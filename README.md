# DCNN_DAE_finetune

1. pyfile folder

1.1  Initial_training.py involves the network building and training.

1.2 finetune.py involves the transfer learning process by fine tuning the layers of DANetV1 using simulation data set.

1.3 test_model.py involves the predicting process using the trained networks DANetV1 or DANetV2.

2. mflie folder

2.1 Data_generating_by_DAE.m involves the process of generating big data set (20000 image pairs in the paper) using DAE model, a physical model mapping the near-field intensity to the antenna surface deformation.

2.2 plot_loss_curves.m involves the loss curves plotting.

2.3 Cal_diffx_2d.m and Cal_diffy_2d.m are dependent files.

3. Network_model folder

These two .h5 files are the trained models saved after initial training and fine tuning, respectively.

4. Simulation

4.1 The .m files file in the grasp_batch folder are the parallel simulation files using GRASP, a antenna near-field simulation software. And the .grd file in this folder is the dependent data to calculate NFIR in the paper.

4.2 The validation folder involves 3 simulation results used as the validation data in the results part of the paper.

5. Validation folder

This folder contains 12 images of the validation results including the corresponding NFIR inputs, deformation labels (the ground truth), the DANetV1 predicting results and the DANetV2 predicting results.


The data sets used to train or fine tune the networks are too large to upload. Please contact me to send by other means if necessary.
