function tensor = ncon(tensorList,legLinks,sequence,finalOrder)
% ncon v1.01 (c) R. N. C. Pfeifer, 2014.
% ==========
% Network CONtractor: NCON
% function A = ncon(tensorList,legLinks,sequence,finalOrder)
% Contracts a single tensor network.
% 
% Supports disjoint networks, trivial (dimension 1) indices, 1D objects, traces, and outer products (both through the zero-in-sequence notation and
% through labelling an implicit trailing index of dimension 1).

% v1.01
% =====
% Added ability to disable input checking for faster performance

    if ~exist('finalOrder','var')
        % finalOrder not specified - use default: Negative indices in descending order, consecutive and starting from -1
        finalOrder = [];
    end
    
    % Check inputs, generate default contraction sequence if required
    if ~exist('sequence','var')
        [sequence legLinks] = checkInputs(tensorList,legLinks,finalOrder);
    else
        [sequence legLinks] = checkInputs(tensorList,legLinks,finalOrder,sequence);
    end
    
    if ~isempty(finalOrder)
        % Apply final ordering request
        legLinks = applyFinalOrder(legLinks,finalOrder);
    end
    
    [tensor legs] = performContraction(tensorList,legLinks,sequence);
    tensor = tensor{1};
    legs = legs{1};
    
    % Arrange legs of final output
    if numel(legs)>1 && ~isequal(legs,-1:-1:numel(legs))
        perm(-legs) = 1:numel(legs);
        tensor = permute(tensor,perm);
    end
end

function legLinks = applyFinalOrder(legLinks,finalOrder)
    % Applies final leg ordering
    for a=1:numel(legLinks)
        for b=find(legLinks{a}<0)
            legLinks{a}(b) = -find(finalOrder==legLinks{a}(b),1);
        end
    end
end

function [tensorList legLinks] = performContraction(tensorList,legLinks,sequence)
    % Performs tensor contraction
    
    warnedLegs = []; % Legs for which a warning has been generated
    while numel(tensorList)>1 || any(legLinks{1}>0)
        % Ensure contraction sequence is not empty - converts implicit outer products into zeros-in-sequence outer products
        if isempty(sequence)
            sequence = zeros(1,numel(tensorList)-1);
        end
        % Check first entry in contraction sequence
        if sequence(1)==0
            % It's a zero: Perform an outer product according to the rules of zeros-in-sequence notation and update contraction sequence
            [tensorList legLinks sequence warnedLegs] = zisOuterProduct(tensorList,legLinks,sequence,warnedLegs);
        else
            % It's a number: Identify and perform tensor contraction
            % Find the tensors on which this index appears
            tensors = zeros(1,2);
            for a=1:numel(legLinks)
                if any(legLinks{a}==sequence(1))
                    tensors(1+(tensors(1)~=0)) = a;
                end
            end
            if tensors(2)==0
                % Index appears on one tensor only: It's a trace
                % Find all traced indices on this tensor
                tracedIndices = sort(legLinks{tensors(1)});
                tracedIndices = tracedIndices([tracedIndices(1:end-1)==tracedIndices(2:end) false]);
                % Check which traced indices actually appear at the beginning of the sequence. Update contraction list.
                [doingTraces sequence] = findInSequence(tracedIndices,sequence,tensorList,legLinks,tensors);
                if ~isequal(sort(doingTraces),sort(tracedIndices))
                    warnedLegs = warn_suboptimal(doingTraces,tracedIndices,0,warnedLegs,legLinks{tensors(1)},[size(tensorList{tensors(1)}) ones(1,numel(legLinks{tensors(1)})-ndims(tensorList{tensors(1)}))]);
                end
                % Perform traces
                tensorList{tensors(1)} = doTrace(tensorList{tensors(1)},legLinks{tensors(1)},doingTraces);
                % Update leg list
                for a=1:numel(doingTraces)
                    legLinks{tensors(1)}(legLinks{tensors(1)}==doingTraces(a)) = [];
                end
            else
                % Index appears on two tensors: It's a contraction
                % Find all indices common to the tensors being contracted
                commonIndices = legLinks{tensors(1)};
                for a=numel(commonIndices):-1:1
                    if ~any(legLinks{tensors(2)}==commonIndices(a))
                        commonIndices(a) = [];
                    end
                end
                % Check which contracted indices actually appear at the beginning of the sequence. Update contraction list.
                [contractionIndices sequence] = findInSequence(commonIndices,sequence,tensorList,legLinks,tensors);
                if ~isequal(sort(contractionIndices),sort(commonIndices))
                    tdims = [size(tensorList{tensors(1)}) ones(1,numel(legLinks{tensors(1)})-ndims(tensorList{tensors(1)}))];
                    tdims = tdims(1:numel(legLinks{tensors(1)}));
                    tdims = [tdims size(tensorList{tensors(2)}) ones(1,numel(legLinks{tensors(2)})-ndims(tensorList{tensors(2)}))]; %#ok<AGROW>
                    warnedLegs = warn_suboptimal(contractionIndices,commonIndices,1,warnedLegs,[legLinks{tensors(1)} legLinks{tensors(2)}],tdims);
                end
                % Are there any (non-trivial) traced indices on either of these tensors? If so, warn sequence is suboptimal
                traces1 = sort(legLinks{tensors(1)});
                traces1 = traces1([traces1(1:end-1)==traces1(2:end) false]);
                traces2 = sort(legLinks{tensors(2)});
                traces2 = traces2([traces2(1:end-1)==traces2(2:end) false]);
                if ~isempty([traces1 traces2])
                    tdims = [size(tensorList{tensors(1)}) ones(1,numel(legLinks{tensors(1)})-ndims(tensorList{tensors(1)}))];
                    tdims = tdims(1:numel(legLinks{tensors(1)}));
                    tdims = [tdims size(tensorList{tensors(2)}) ones(1,numel(legLinks{tensors(2)})-ndims(tensorList{tensors(2)}))]; %#ok<AGROW>
                    warnedLegs = warn_suboptimal(contractionIndices,[traces1 traces2],2,warnedLegs,[legLinks{tensors(1)} legLinks{tensors(2)}],tdims);
                end
                % Contract over these indices and update leg list
                [tensorList{tensors(1)} legLinks{tensors(1)}] = tcontract(tensorList{tensors(1)},tensorList{tensors(2)},legLinks{tensors(1)},legLinks{tensors(2)},contractionIndices);
                tensorList(tensors(2)) = [];
                legLinks(tensors(2)) = [];
            end
        end
    end
end

function [rtnIndices sequence] = findInSequence(indices,sequence,tensorList,legLinks,tensors)
    % Check how many of the supplied indices appear at the beginning of "sequence" - these are the indices to return
    ptr = 1;
    while ptr<=numel(sequence) && any(indices==sequence(ptr))
        ptr = ptr + 1;
    end
    rtnIndices = sequence(1:ptr-1);
    % If not contracting all possible non-trivial indices at once, warn that sequence is suboptimal
    % - remove uncontracted trivial indices from comparison list as postponing these is unimportant
    for a=numel(indices):-1:1
        if ~any(rtnIndices==indices(a)) && size(tensorList{tensors(1)},find(legLinks{tensors(1)}==indices(a),1))==1
            indices(a) = []; % Not doing this trace yet, but is trivial so postponing it is not a concern
        end
    end
    % Update contraction sequence
    sequence = sequence(ptr:end);
end

function B = doTrace(A,legLabels,tracedIndices)
    % Trace over all indices listed in tracedIndices, each of which occurs twice on tensor A
    sz = size(A);
    sz = [sz ones(1,numel(legLabels)-numel(sz))];
    tpos = [];
    % Find positions of tracing indices
    for a=1:numel(tracedIndices)
        tpos = [tpos find(legLabels==tracedIndices(a))]; %#ok<AGROW>
    end
    % Reorder list of tracing index positions so that they occur in two equivalent blocks
    sztrace = prod(sz(tpos(1:2:end)));
    tpos = [tpos(1:2:end) tpos(2:2:end)];
    % Identify non-tracing index positions
    ind = 1:numel(legLabels);
    ind(tpos) = [];
    % Collect non-tracing and tracing indices
    A = reshape(permute(A,[ind tpos]),prod(sz(ind)),sztrace,sztrace); % Separate indices to be traced and not to be traced
    B = 0;
    % Perform trace
    for a=1:sztrace
        B = B + A(:,a,a); % Perform trace
    end
    B = reshape(B,[sz(ind) 1 1]);
end

function [tensor legs] = tcontract(T1,T2,legs1,legs2,contractLegs)
    % Contract T1 with T2 over indices listed in contractLegs
    
    % If either tensor is a number (no legs), add a trivial leg to contract over.
    if numel(legs1)==0
        legs1 = max(abs(legs2))+1;
        legs2 = [legs2 legs1];
        contractLegs = legs1;
    else
        if numel(legs2)==0
            legs2 = max(abs(legs1))+1;
            legs1 = [legs1 legs2];
            contractLegs = legs2;
        end
    end
    
    % Find uncontracted legs
    freeLegs1 = legs1;
    freeLegs2 = legs2;
    posFreeLegs1 = 1:numel(legs1);
    posFreeLegs2 = 1:numel(legs2);
    for a=1:numel(contractLegs)
        posFreeLegs1(freeLegs1==contractLegs(a)) = [];
        freeLegs1(freeLegs1==contractLegs(a)) = [];
        posFreeLegs2(freeLegs2==contractLegs(a)) = [];
        freeLegs2(freeLegs2==contractLegs(a)) = [];
    end
    
    % Find contracted legs; match ordering of contracted legs on tensors T1 and T2
    posContLegs1 = 1:numel(legs1);
    posContLegs1(posFreeLegs1) = [];
    posContLegs2 = zeros(1,numel(posContLegs1));
    for a=1:numel(posContLegs1)
        posContLegs2(a) = find(legs2==legs1(posContLegs1(a)),1);
    end
    
    sz1 = [size(T1) ones(1,numel(legs1)-ndims(T1))];
    sz2 = [size(T2) ones(1,numel(legs2)-ndims(T2))];
    if numel(legs1)>1
        T1 = permute(T1,[posFreeLegs1 posContLegs1]);
    end
    if numel(legs2)>1
        T2 = permute(T2,[posContLegs2 posFreeLegs2]);
    end
    linkSize = prod(sz1(posContLegs1)); % NB prod([]) = 1 if no contracted legs
    T1 = reshape(T1,prod(sz1(posFreeLegs1)),linkSize);
    T2 = reshape(T2,linkSize,prod(sz2(posFreeLegs2)));
    tensor = T1 * T2;
    tensor = reshape(tensor,[sz1(posFreeLegs1) sz2(posFreeLegs2) 1 1]);
    
    % Return uncontracted index list. Uncontracted legs are in order [unrearranged uncontracted legs off tensor 1, unrearranged uncontracted legs off tensor 2].
    legs = [legs1(posFreeLegs1) legs2(posFreeLegs2)];
end

function warnedLegs = warn_suboptimal(doing,couldDo,mode,warnedLegs,legList,legDims)
    % Generate warning for detected suboptimal contraction sequence
    % Mode 0: Doing traces on a tensor, did not do all at once
    % Mode 1: Contracting two tensors, missed some connecting legs
    % Mode 2: Contracting two tensors, one carries a traced index which has not yet been evaluated
    
    % Let couldDo be the list of indices which should be contracted but which weren't
    for a=1:numel(doing)
        couldDo(couldDo==doing(a)) = [];
    end
    % Check if warning has already been generated for these legs
    for a=1:numel(warnedLegs)
        couldDo(couldDo==warnedLegs(a)) = [];
    end
    % Check if legs are trivial (do not warn for trivial legs as the contraction of these is unimportant)
    for a=numel(couldDo):-1:1
        if legDims(find(legList==couldDo(a),1))==1
            warnedLegs = [warnedLegs couldDo(a)]; %#ok<AGROW>
            couldDo(a) = [];
        end
    end
    if ~isempty(couldDo)
        if mode == 2
            t = 'Sequence suboptimal: Before contracting over ind';
            if numel(doing)==1
                t = [t 'ex ' num2str(doing) ' please trace over ind'];
            else
                t = [t 'ices ' num2str(doing) ' please trace over ind'];
            end
            if numel(couldDo)==1
                t = [t 'ex ' num2str(couldDo) '.'];
            else
                t = [t 'ices ' num2str(couldDo) '.'];
            end
        else
            if ~isempty(doing)
                t = 'Sequence suboptimal: When contracting ind';
                if numel(doing)==1
                    t = [t 'ex ' num2str(doing) ' please also contract ind'];
                else
                    t = [t 'ices ' num2str(doing) ' please also contract ind'];
                end
                if numel(couldDo)==1
                    t = [t 'ex ' num2str(couldDo) ' as these indices appear on the same '];
                else
                    t = [t 'ices ' num2str(couldDo) ' as these indices connect the same '];
                end
                if mode == 0
                    t = [t 'tensor.'];
                else
                    t = [t 'two tensors.'];
                end
            else
                t = 'Sequence suboptimal: Instead of performing an outer product and tracing later, please contract ind';
                if numel(couldDo)==1
                    t = [t 'ex ' num2str(couldDo) '. This index connects the same two tensors and is non-trivial.'];
                else
                    t = [t 'ices ' num2str(couldDo) '. These indices connect the same two tensors and are non-trivial.'];
                end
            end
        end
        warning('ncon:suboptimalsequence',t);
        warnedLegs = [warnedLegs couldDo];
    end
end

function [sequence legLinks] = checkInputs(tensorList,legLinks,finalOrder,sequence)
    % Checks format of input data and returns separate lists of positive and negative indices
    
    global ncon_skipCheckInputs;
    if isequal(ncon_skipCheckInputs,true)
        for a=1:numel(legLinks)
            if isempty(legLinks{a})
                legLinks{a} = zeros(1,0);
            end
        end
        if ~exist('sequence','var')
            sequence = cell2mat(legLinks);
            sequence = sort(sequence(sequence>0));
            sequence = sequence(1:2:end);
        end
    else
        % Check data sizes
        if size(tensorList,1)~=1 || size(tensorList,2)~=numel(tensorList)
            error('Array of tensors has incorrect dimension - should be 1xn')
        end
        if ~isequal(size(legLinks),size(tensorList))
            error('Array of links should be the same size as the array of tensors')
        end
        for a=1:numel(legLinks)
            if size(legLinks{a},1)~=1 || size(legLinks{a},2)~=numel(legLinks{a})
                if isempty(legLinks{a})
                    legLinks{a} = zeros(1,0);
                else
                    error(['Leg link entry ' num2str(a) ' has wrong dimension - should be 1xn']);
                end
            end
            tsize = size(tensorList{a});
            if numel(tsize)==2 && tsize(2)==1
                tsize = tsize(tsize~=1);
            end
            if numel(legLinks{a}) < numel(tsize)
                if numel(legLinks{a})==1
                    error(['Leg link entry ' num2str(a) ' is too short: Tensor size is [' num2str(size(tensorList{a})) '] and legLinks{' num2str(a) '} has only ' num2str(numel(legLinks{a})) ' entry.']);
                else
                    error(['Leg link entry ' num2str(a) ' is too short: Tensor size is [' num2str(size(tensorList{a})) '] and legLinks{' num2str(a) '} has only ' num2str(numel(legLinks{a})) ' entries.']);
                end
            end
        end
        % Check all tensors are numeric
        for a=1:numel(tensorList)
            if ~isnumeric(tensorList{a})
                warning('Tensor list must be a 1xn cell array of numerical objects')
            end
        end
        % If finalOrder is provided, check it is a list of unique negative integers
        if ~isempty(finalOrder)
            if ~isnumeric(finalOrder)
                error('finalOrder must be a list of unique negative integers')
            elseif any(imag(finalOrder)~=0) || any(real(finalOrder)>0)
                error('finalOrder must be a list of unique negative integers')
            end
            t1 = sort(finalOrder,'descend');
            if any(t1(1:end-1)==t1(2:end))
                error('finalOrder must be a list of unique negative integers')
            end
        end
        % Get list of positive indices
        allindices = cell2mat(legLinks);
        if any(allindices==0)
            error('Zero entry in legLinks')
        elseif any(imag(allindices)~=0)
            error('Complex entry in legLinks')
        elseif any(int32(allindices)~=allindices)
            error('Non-integer entry in legLinks');
        end
        [posindices ix] = sort(allindices(allindices>0),'ascend');
        % Test all positive indices occur exactly twice
        if mod(numel(posindices),2)~=0
            maxposindex = posindices(end);
            posindices = posindices(1:end-1);
        end
        flags = (posindices(1:2:numel(posindices))-posindices(2:2:numel(posindices)))~=0;
        if any(flags)
            errorpos = 2*find(flags~=0,1,'first')-1;
            if errorpos>1 && posindices(errorpos-1)==posindices(errorpos)
                error(['Error in index list: Index ' num2str(posindices(errorpos)) ' appears more than twice']);
            else
                error(['Error in index list: Index ' num2str(posindices(errorpos)) ' only appears once']);
            end
        end
        if exist('maxposindex','var')
            if isempty(posindices)
                error(['Error in index list: Index ' num2str(maxposindex) ' only appears once']);
            end
            if posindices(end)==maxposindex
                error(['Error in index list: Index ' num2str(maxposindex) ' appears more than twice']);
            else
                error(['Error in index list: Index ' num2str(maxposindex) ' only appears once']);
            end
        end
        altposindices = posindices(1:2:numel(posindices));
        flags = altposindices(1:end-1)==altposindices(2:end);
        if any(flags)
            errorpos = find(flags,1,'first');
            error(['Error in index list: Index ' num2str(altposindices(errorpos)) ' appears more than twice']);
        end
        % Check positive index sizes match
        sizes = ones(size(allindices));
        ptr = 1;
        for a=1:numel(tensorList)
            sz = size(tensorList{a});
            if numel(legLinks{a})==1 % Is a vector (1D)
                sz = max(sz);
            end
            sizes(ptr:ptr+numel(sz)-1) = sz;
            ptr = ptr + numel(legLinks{a});
        end
        sizes = sizes(allindices>0); % Remove negative legs
        sizes = sizes(ix); % Sort in ascending positive leg sequence
        flags = sizes(1:2:end)~=sizes(2:2:end);
        if any(flags)
            errorpos = find(flags,1,'first');
            error(['Leg size mismatch on index ' num2str(altposindices(errorpos))]);
        end
        % Check negative indices are unique and consecutive, or unique and correspond to entries in finalOrder
        negindices = sort(allindices(allindices<0),'descend');
        if any(negindices(1:end-1)==negindices(2:end))
            error('Negative indices must be unique');
        end
        if isempty(finalOrder)
            if ~isequal(negindices,-1:-1:-numel(negindices))
                error('If finalOrder is not specified, negative indices must be consecutive starting from -1');
            end
        else
            if ~isequal(negindices,sort(finalOrder,'descend'))
                error('Negative indices must match entries in finalOrder')
            end
        end
        if exist('sequence','var')
            % Check sequence is a row vector of positive real integers, each occurring only once, and zeros.
            % Check they match the positive leg labels.
            if any(uint32(sequence)~=sequence)
                error('All entries in contraction sequence must be real positive integers or zero');
            end
            if numel(altposindices)~=sum(sequence>0)
                error('Each positive index must appear once and only once in the contraction sequence, and each index in the sequence must appear on the tensors.');
            end
            if ~isempty(altposindices)
                if any(altposindices~=sort(sequence(sequence>0)))
                    error('Each positive index must appear once and only once in the contraction sequence');
                end
            end
        else
            sequence = altposindices;
        end
    end
    if numel(sequence)==0
        sequence = zeros(1,0);
    end
end

function [tensorList legLinks sequence warnedLegs] = zisOuterProduct(tensorList,legLinks,sequence,warnedLegs)
    % This function provides support for the zeros-in-sequence notation described in arXiv:1304.6112

    % Perform one or more outer products described by zeros in the contraction sequence
    
    if all(sequence==0) % Final outer product of all remaining objects - ensure enough zeros are present in the sequence
        if numel(sequence) < numel(legLinks)-1
            sequence = zeros(1,numel(legLinks)-1);
            warning('ncon:zisShortSequence','Zeros-in-sequence notation used, and insufficient zeros provided to describe final tensor contraction. Finishing contraction anyway.');
        end
    end
    
    % Determine number of outer products pending
    numOPs = 1;
    while sequence(numOPs)==0 && numOPs < numel(sequence)
        numOPs = numOPs + 1;
    end
    if sequence(numOPs)~=0
        numOPs = numOPs - 1;
    end
    % Determine list of tensors on which OP is to be performed
    if numOPs == numel(legLinks)-1
        % OP of all remaining tensors
        OPlist = 1:numel(legLinks);
    else
        % For OP of n tensors (n=numOPs+1) when more than n tensors remain, proceed past the zeros in the sequence and read nonzero indices until 
        % n+1 tensors accounted for. Failure to find n+1 tensors implies an invalid sequence.
        flags = false(1,numel(legLinks));
        ptr = numOPs+1;
        while sum(flags) < numOPs+2
            % Flag tensors on which leg given by sequence(ptr) appears
            if ptr > numel(sequence)
                t = 'Contraction sequence includes zeros and is inconsistent with rules of zeros-in-sequence notation. After a ';
                if numOPs==1
                    t = [t 'zero']; %#ok<AGROW>
                else
                    t = [t 'string of ' num2str(numOPs) ' zeros']; %#ok<AGROW>
                end
                error([t ', while reading further indices to identify the ' num2str(numOPs+1) ' tensors involved in the outer product, ncon encountered end of index list before identifying all tensors.']);
            end
            if sequence(ptr)==0
                t = 'Contraction sequence includes zeros and is inconsistent with rules of zeros-in-sequence notation. After a ';
                if numOPs==1
                    t = [t 'zero']; %#ok<AGROW>
                else
                    t = [t 'string of ' num2str(numOPs) ' zeros']; %#ok<AGROW>
                end
                error([t ', while reading further indices to identify the ' num2str(numOPs+1) ' tensors involved in the outer product, ncon encountered another zero before identifying all tensors.']);
            end
            count = 0;
            for a=1:numel(legLinks)
                if any(legLinks{a}==sequence(ptr))
                    flags(a) = true;
                    count = count + 1;
                end
            end
            if count~=2
                t = 'Contraction sequence includes zeros and is inconsistent with rules of zeros-in-sequence notation. After a ';
                if numOPs==1
                    t = [t 'zero']; %#ok<AGROW>
                else
                    t = [t 'string of ' num2str(numOPs) ' zeros']; %#ok<AGROW>
                end                
                error([t ', while reading further indices to identify the ' num2str(numOPs+1) ' tensors involved in the outer product, ncon encountered an index ' num2str(sequence(ptr)) ' which appears on ' num2str(count) ' tensor(s). Index should appear on exactly 2 tensors at this time.']);
            end
            ptr = ptr + 1;
        end
        % Identify which of these tensors is _not_ participating in the OP (but is instead contracted with the result of the OP), and unflag it.
        % - Identify the two tensors on which the first nonzero index appears
        % - Examine consecutive nonzero indices until one matches only one of the two tensors. This is the tensor to unflag.
        firsttensors = [0 0];
        ptr = numOPs+1;
        for a=1:numel(legLinks)
            if any(legLinks{a}==sequence(ptr))
                if firsttensors(1)==0
                    firsttensors(1) = a;
                else
                    firsttensors(2) = a;
                    break;
                end
            end
        end
        done = false;
        while ~done
            nexttensors = [0 0];
            ptr = ptr + 1;
            for a=1:numel(legLinks)
                if any(legLinks{a}==sequence(ptr))
                    if nexttensors(1)==0
                        nexttensors(1) = a;
                    else
                        nexttensors(2) = a;
                        break;
                    end
                end
            end
            if ~isequal(firsttensors,nexttensors)
                done = true;
            end
        end
        if any(firsttensors == nexttensors(1))
            postOPtensor = nexttensors(1);
        else
            postOPtensor = nexttensors(2);
        end
        flags(postOPtensor) = false;
        OPlist = find(flags);
        % - Check contraction with postOPtensor is over all non-trivial indices of OP tensors
        OPindices = cell2mat(legLinks(OPlist));
        for a=1:numel(OPindices)
            if ~any(legLinks{postOPtensor}==OPindices(a))
                isnontriv = true;
                for b=1:numel(OPlist)
                    if any(legLinks{b}==OPindices(a))
                        isnontriv = size(tensorList{b},legLinks{b}(find(legLinks{b}==OPindices(a),1)))~=1;
                        break;
                    end
                end
                if isnontriv
                    error(['Contraction sequence includes zeros and is inconsistent with rules of zeros-in-sequence notation. After using zeros to contract a group of tensors, all non-trivial indices on those tensors must be contracted with the next object. Contraction did not include index ' num2str(OPindices(a)) '.']);
                end
            end
        end
    end
    % Find sizes of all tensors involved in OP.
    OPsizes = zeros(1,numel(OPlist));
    for a=1:numel(OPlist)
        OPsizes(a) = numel(tensorList{OPlist(a)});
    end
    % Perform OPs
    while numel(OPsizes)>1
        % Find smallest two tensors
        [~, ix] = sort(OPsizes,'ascend');
        % If they have common nontrivial indices, warn about suboptimal sequence
        commonIndices = legLinks{OPlist(ix(1))};
        for a=numel(commonIndices):-1:1
            if ~any(legLinks{OPlist(ix(2))}==commonIndices(a))
                commonIndices(a) = [];
            else
                if size(tensorList{OPlist(ix(1))},find(legLinks{OPlist(ix(1))}==commonIndices(a),1))==1
                    commonIndices(a) = [];
                end
            end
        end
        if ~isempty(commonIndices)
            % Suboptimal contraction sequence - generate warning
            tdims = [size(tensorList{OPlist(ix(1))}) ones(1,numel(legLinks{OPlist(ix(1))})-ndims(tensorList{OPlist(ix(1))}))];
            tdims = tdims(1:numel(legLinks{OPlist(ix(1))}));
            tdims = [tdims size(tensorList{OPlist(ix(2))}) ones(1,numel(legLinks{OPlist(ix(2))})-ndims(tensorList{OPlist(ix(2))}))]; %#ok<AGROW>
            warnedLegs = warn_suboptimal([],commonIndices,1,warnedLegs,[legLinks{OPlist(ix(1))} legLinks{OPlist(ix(2))}],tdims);
        end
        % Contract them
        [tensorList{OPlist(ix(1))} legLinks{OPlist(ix(1))}] = tcontract(tensorList{OPlist(ix(1))},tensorList{OPlist(ix(2))},legLinks{OPlist(ix(1))},legLinks{OPlist(ix(2))},[]);
        tensorList(OPlist(ix(2))) = [];
        legLinks(OPlist(ix(2))) = [];
        OPsizes(ix(1)) = OPsizes(ix(1)) * OPsizes(ix(2));
        OPsizes(ix(2)) = [];
        OPlist(OPlist>OPlist(ix(2))) = OPlist(OPlist>OPlist(ix(2))) - 1;
        OPlist(ix(2)) = [];
    end
    % Update sequence
    sequence = sequence(numOPs+1:end);
end
