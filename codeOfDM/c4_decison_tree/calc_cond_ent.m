function cond_ent=calc_cond_ent(M,f_id)
% input :data and the feature_id which to divide by 
G=findgroups(M(:,f_id)); % divide the date by the label of feature 1
% max(G);
cond_ent=0;
for i=1:max(G)
    Di=M(G==i,:); % partial  data
    cond_ent=cond_ent+calc_ent(Di)*size(Di,1)/size(M,1);
end


end
