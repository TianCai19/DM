function ent = calc_ent(M)
%  input: a table contains the data (which last column store the classfication)
%  output:the entropy of the data

%   class=M(:,end) % get the classification
    cla=M(:,end).Variables; % turn it into categorical
    cla=categorical(cla);
    counts=countcats(cla);% get the count 
    p=counts/sum(counts);% get the probility distribution
    % calculate the entropy
%     -p.*log2(p);
    ent=sum(-p.*log2(p));
end
