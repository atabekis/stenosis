# callbacks.py

# Python imports
import sys
import time
import socketserver
import threading

import pytorch_lightning as pl

# Local imports
from util import log
from config import TEST_MODEL_ON_KEYBOARD_INTERRUPT, REMOTE_TEST_PORT, REMOTE_TEST_COMMAND



class TestOnKeyboardInterruptCallback(pl.Callback):
    """
    Helper callback to test the model on KeyboardInterrupt.
    """
    def on_exception(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", exception: Exception) -> None:
        """
        when an exception occurs in training, validation, or testing loop.
        """
        if isinstance(exception, KeyboardInterrupt) and TEST_MODEL_ON_KEYBOARD_INTERRUPT:
            log("TEST_MODEL_ON_KEYBOARD_INTERRUPT=True. Testing model.")

            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback and hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
                ckpt_to_test = checkpoint_callback.best_model_path
                log(f"Using best model checkpoint for testing: {ckpt_to_test}")

            else:
                log("No best_model_path found testing with 'last' checkpoint (if available) or current model weights.")
                ckpt_to_test = "last"


            log(f"Starting test with checkpoint: {ckpt_to_test if ckpt_to_test else 'current model weights'}.")
            trainer.test(model=pl_module, datamodule=trainer.datamodule, ckpt_path=ckpt_to_test)

        raise exception # always exit, PL handles graceful shutdown



class RemoteTestTriggerCallback(pl.Callback):
    """
    Helper callback to trigger testing via a network command. This functionality is used in HPC where
    keyboard interrupting is not possible. Program terminates after testing.

    Use case (after SSHing into the server): "echo "REMOTE_TEST_COMMAND" | nc localhost REMOTE_TEST_PORT"
        My code uses the commands from config.py: 'echo TEST | nc localhost 3131'
    """
    def __init__(self,
                 port: int = REMOTE_TEST_PORT,
                 command: str = REMOTE_TEST_COMMAND):
        super().__init__()
        self.port = port
        self.command = command.strip()
        self._trigger = threading.Event()
        self._server: socketserver.ThreadingTCPServer | None = None
        self._thread: threading.Thread | None = None


    class _Handler(socketserver.StreamRequestHandler):
        def handle(self):
            # read line coming from connection
            line = self.rfile.readline().decode('utf-8', 'ignore').strip()
            log(f"Connection from {self.client_address}: '{line}'")

            if line == self.server.callback.command:
                self.server.callback._trigger.set()
                self.wfile.write(b"ACK: Test will run and program will exit.\n")

            else:
                log(f"Unknown command: '{line}'")
                self.wfile.write(b"NACK: Unknown command.\n")

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not getattr(trainer, 'is_global_zero', True): # open socket only on rank:0 process
            return

        log(f'Listening on port {self.port} for "{self.command}"')
        self._trigger.clear()
        self._server = socketserver.ThreadingTCPServer(  # have to ssh into local
            ('0.0.0.0', self.port),
            type(self)._Handler
        )
        self._server.callback = self
        self._server.daemon_threads = True
        self._server.allow_reuse_address = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if not getattr(trainer, 'is_global_zero', True) or not self._trigger.is_set():
            return

        # stop listening
        self._server.shutdown()
        self._thread.join(timeout=2)
        self._trigger.clear()

        # select checkpoint
        cb = getattr(trainer, 'checkpoint_callback', None)
        ckpt = getattr(cb, 'best_model_path', None) or getattr(cb, 'last_model_path', None)

        if ckpt: log(f"Using checkpoint: {ckpt}")
        else: log("No checkpoint; using current weights.")

        # run testing
        if trainer.datamodule:
            pl_module.eval()
            exit_code = 0
            try:
                trainer.test(model=pl_module, datamodule=trainer.datamodule, ckpt_path=ckpt)
            except Exception as e:
                log(f"Error during test: {e}")
                exit_code = 1
            finally:
                if pl_module.training: # passing it onto PL to safely exit as well
                    pl_module.train()
        else:
            log("No DataModule; cannot test.")
            exit_code = 1

        log(f"Exiting program with code {exit_code}.")
        sys.exit(exit_code)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._server: # cleanup
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: Exception) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None