clc;clear;
%% 
global frequency distance gain PATH sim_name;
frequency = 0.3; 
distance = 33; 
gain = 0.5; 
PATH = "D:\\GRASP9\\grasp9.exe"; 
sim_name = 'SecondTry'; 
num_of_simulation = 1; 
%% 
for index = 1:num_of_simulation
    GraspFunc_WriteTCI(index);
    GraspFunc_WriteTOR(index);
    GraspFunc_WriteSFC(index);
    GraspFunc_WriteBAT(index);
end
%%
filename = sim_name;
parfor index = 1:num_of_simulation
    dos([filename num2str(index) '.bat']);
end
%% 
for index = 1:num_of_simulation
    GraspFunc_ReadGRD(index);
end
%% 
for index = 1:num_of_simulation
    new_folder = ['./Simulation_No' num2str(index)];
    mkdir(new_folder);
    try 
        movefile([sim_name num2str(index) '.bat'], new_folder, 'f');
        movefile([sim_name num2str(index) '.log'], new_folder, 'f');
        movefile([sim_name num2str(index) '.out'], new_folder, 'f');
        movefile([sim_name num2str(index) '.tci'], new_folder, 'f');
        movefile([sim_name num2str(index) '.tor'], new_folder, 'f');
        movefile([sim_name num2str(index) '.grd'], new_folder, 'f');
        movefile(['po_' num2str(index)], new_folder, 'f');
        movefile(sprintf('Surface%05d.sfc',index), new_folder, 'f');
    end
end
new_folder = './NetworkData';
mkdir(new_folder);
for index = 1:num_of_simulation
    try
        movefile(sprintf('NNinput%05d.png',index), new_folder, 'f');
        movefile(sprintf('NNlabel%05d.png',index), new_folder, 'f');
    end
end