function gain = calc_gain(M,f_i)
    gain=calc_ent(M)-calc_cond_ent(M,f_i);
end

