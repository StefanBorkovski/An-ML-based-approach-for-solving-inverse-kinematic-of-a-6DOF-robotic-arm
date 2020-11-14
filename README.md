## Project explanation

In this work, a machine learning-based approach for solving the inverse kinematic of a robotic arm with six degrees of freedom is presented. For comparison, a solution for a robotic arm with three degrees of freedom is shown. This project is part of my bachelor thesis work.

Solving the inverse kinematics of a robotic arm is a necessary step whether it is a first-time installation or the configuration of the robotic arm is already changed.

Today there are many analytical ways for solving the inverse kinematics and sometimes they can be hard or even impossible because of the complexity of the system. As a consequence, developing new methodologies for solving the inverse kinematic is part of today's research in the field of robotics. The complexity of solving the inverse kinematic can increase corresponding with the construction complexity of the robotic arm as well as the high nonlinearity property of the equations that are describing the joint dependancies. Analytical generated equations for inverse kinematics sometimes do not describe the dynamic of the real arm that is being identified and for that reason, additional calibrating is required. Three methodologies are presented for comparison. The neural-network-based approach is chosen as the best method. It is trained on data that is generated with the help of direct kinematic. The model can be continuously retrained with new data and can serve as an automated calibration method in contrast with the classical calibration.

Nowadays the application of artificial intelligence in the industry is starting to accelerate. There is an enormous research going on in this field which is continuously resulting in solutions that are helping humanity. This work is an example showing the power of these algorithms, able for learning strong nonlinearities. The achieved results are broadening the opportunities for usage of methods like these in robotic arm kinematics.

The use of machine learning-based techniques in robotic arm kinematics can enable online calibration of the robotic arms. With the use of this algorithm, the robotic arms can become independent, to some degree, from the changes in the environment such as small movement caused by vibrations, changes in the configuration of the robotic arm because of mechanical parts wear, etc... This suggests that there won't be a need for stopping the robotic arm for calibration which can cause great financial losses.

▸ This work is done as a project for my bachelor's degree. June 2019.

## Scripts explanation

### “Gradient_Boosting_Regressor.py” 
	This code is used for training and testing the Gradient Boosting Regressor-based model for 3 and 6 DOF.
### “Forward_Kinematics_DH_Test.py”
	This code is used for calculating the final transformation matrix using the DH convention.
### “3D_plot_coordinates.py”
	This code is used for visualization of the created data set in 3D. It also represents the working environment of the robotic arm.
### “LSTM_Neural_Network.py”
	This code is used for training and testing the LSTM neural network-based model for 3 and 6 DOF.
### “FC_Accuracy_Test_3DOF.py”
	This code is used for 3D visualization of training and test points for the robotic arm with 3 DOF.
### “Sequential_Neural_Network.py”
	This code is used for training and testing the Sequential Neural Network based model for 3 and 6 DOF.
### “FC_Accuracy_Test_6DOF.py”
	This code is used for 3D visualization of training and test points for the robotic arm with 6 DOF.
### “merging_data.py”
	This code is used for merging two data sets. Also for saving the newly created dataset.
### “dataset_generation_6_wrists”
	This code is used for creating data set for the 6DOF robotic arm with the use of direct kinematic.
### “removing_duplicates.py”
	This code is used to clear the dataset from duplicates (points that are very close). This method proved that it is improving the training phase.
### “dataset_generation_3_wrists”
	This code is used for creating data set for the 3DOF robotic arm with the use of direct kinematic.
### “adding_features.py”
	This code is used for generating new features as radius vector, end-effector orientation, etc…



