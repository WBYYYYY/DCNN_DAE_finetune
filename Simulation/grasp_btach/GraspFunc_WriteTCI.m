function GraspFunc_WriteTCI(index)
    global sim_name;
    % tic文件内容如下：作用是grasp9里面的运行指令，注意修改输出到Field1，也就是预先定义好的表面
    % FILES READ ALL FirstTry.tor
    % COMMAND OBJECT PO_Calc_1 get_currents (source:sequence(ref(Feed_1)), auto_convergence_of_po:on, convergence_on_output_grid:sequence(ref(Field1)))
    % COMMAND OBJECT Field1 get_field (source:ref(PO_Calc_1))
    % QUIT
    fid = fopen([sim_name num2str(index) '.tci'], 'w');
    fprintf(fid, "FILES READ ALL ");
    fprintf(fid, sim_name);
    fprintf(fid, num2str(index));
    fprintf(fid, ".tor");
    fprintf(fid,'\n');
    fprintf(fid,"COMMAND OBJECT PO_Calc_1 get_currents (source:sequence(ref(Feed_1)), auto_convergence_of_po:on, convergence_on_output_grid:sequence(ref(Field1)))");
    fprintf(fid,'\n');
    fprintf(fid,"COMMAND OBJECT Field1 get_field (source:ref(PO_Calc_1))");
    fprintf(fid,'\n');
    fprintf(fid,"QUIT");fprintf(fid,'\n');
    fclose(fid);
end