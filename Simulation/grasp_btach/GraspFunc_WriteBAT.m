function GraspFunc_WriteBAT(index)
    global PATH sim_name;
    fid = fopen([sim_name num2str(index) '.bat'], 'w');
    fprintf(fid, PATH);
    fprintf(fid, " ");
    fprintf(fid, sim_name);
    fprintf(fid, num2str(index));
    fprintf(fid, ".tci ");
    fprintf(fid, sim_name);
    fprintf(fid, num2str(index));
    fprintf(fid, ".log ");
    fprintf(fid, sim_name);
    fprintf(fid, num2str(index));
    fprintf(fid, ".out");
    fclose(fid);
end


