The CADICA dataset is an annotated Invasive Coronary Angiography (ICA) dataset of 42 patients. In ICA imaging, lesion degree assessment is commonly done by visual estimation, which implies a subjective factor and interobserver variability. Accurate recognition of lesions is crucial for a correct diagnosis and treatment. This motivates the development of computer-aided systems that can support specialists in their clinical procedures. This dataset can be used by clinicians to train their skills in angiographic assessment of CAD severity, by computer scientists to create computer-aided diagnostic systems to help in such assessment, and to validate existing methods for CAD detection approaching solutions for clinical settings.

CADICA dataset includes ICA images, manually labeled lesion bounding boxes, and selected clinical features.

The CADICA dataset becomes a directory that contains the "metadata.xlsx" file, which is the file where the clinical data is located, as well as two main folders that differentiate the videos selected by the medical team for each patient: "nonselectedVideos" and "selectedVideos". Inside each folder, there are several sub-directories with the naming convention "pX", where X is the ID of each patient, and "vY", where Y is the ID of the video of that patient. 

The folder "pX" contains the following information: 
	"vY": several sub-directories with the videos selected for that patient. 
	"lesionVideos.txt": includes the IDs of the videos chosen where appears at least one lesion which is labeled.
	"nonlesionVideos.txt": contains the IDs of the selected videos with no visible lesions.

The folder "vY" contains the following information:
	"input": a sub-directory containing a separate PNG file for each video frame.
	"pX_vY_selectedFrames.txt": includes the IDs of the keyframes for the medical team for all the selected videos. 
	"groundtruth": a sub-directory available only if there are lesions in that selected video.

The folder "groundtruth" contains the following information:
	"pX_vY_000ZZ.txt": contains the bounding boxes and their category in each row. There are such files as frames in "pX_vY_selectedFrames.txt". Bounding boxes are specified in the format [x,y,w,h], where (x,y) are the pixel coordinates of the top left corner, w is the width and h is the height of the bounding box.
	"pX_vY_groundTruthTable.mat": contains a table with the ground truth information of that video. 