% Klein, George, Robert E. Lindberg, and Richard W. Longman.
% “Computation of a Degree of Controllability via System Discretization.�?
% Journal of Guidance, Control, and Dynamics 5,
% no. 6 (November 1982): 583–88. https://doi.org/10.2514/3.19793.

function DOC=doc_multicopter(coaxial, rotors, fmax, d, M_uav, M_motor, M_prop, T, N) 
    arguments
        coaxial
        rotors
        fmax  double
        d double
        M_uav double
        M_motor double
        M_prop double
        T double
        N double
    end
    %clear all; clc;
    %format long;
    % Multicopter Example 1----------------------------------------------------
    tic
    %M=2;
    %Ixx=0.0411;
    %Iyy=0.0478;
    %Izz=0.0599;
    %d=0.28;
    j=0.1;
    Grav=[-M_uav*9.81;0;0;0];
    %fmax=6; %max thrust of one propeller
    %config=4;
    %Time variables------------------------------------------------------------
    %T = 0.5; % time
    %N = 2; % nb steps
    M_center=M_uav-4*(M_prop+M_motor);
    
    % Determination of Multirotor Configuration
    
    if coaxial==1;
        if rotors==8;
            config=2;
        elseif rotors==12;
            config=4;
        end
    elseif coaxial==0;
        if rotors==4;
            config=1;
        elseif rotors==6;
            config=3;
        elseif rotors==8;
            config=5;
        end
    end
   
    % Configuration Cases
    
    switch config
        case 1 %Simple Quadcopter
    %         Bf= [1 1 1 1; 
    %             sqrt(2)/2 -sqrt(2)/2 -sqrt(2)/2 sqrt(2)/2; 
    %             sqrt(2)/2 sqrt(2)/2 -sqrt(2)/2 -sqrt(2)/2; 
    %             1 -1 1 -1];
              Bf= [1 1 1 1; 
                   sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 sqrt(2)*d/2; 
                   sqrt(2)*d/2 sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2; 
                   j -j j -j];
            maxinput = [1 1 1 1]'; %max thrust force (N)
            mininput = [0 0 0 0]';
            Ixx=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 4*((M_prop+M_motor)*(sqrt(2)*d/2)^2);
            Iyy=Ixx;
            Izz=M_center*0.5^(2)/2+4*((M_prop+M_motor)*(d^2));
            disp('Simple Quadcopter');
        case 2 %Coaxial Quadcopter
            Bf= [1 1 1 1 1 1 1 1; 
                sqrt(2)*d/2 sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 sqrt(2)*d/2 sqrt(2)*d/2; 
                sqrt(2)*d/2 sqrt(2)*d/2 sqrt(2)*d/2 sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2 -sqrt(2)*d/2; 
                j -j j -j -j j j -j];
            maxinput = [1 1 1 1 1 1 1 1]'; %max thrust force (N)
            mininput = [0 0 0 0 0 0 0 0]';
            Ixx=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 8*((M_prop+M_motor)*(sqrt(2)*d/2)^2);
            Iyy=Ixx;
            Izz=M_center*0.5^(2)/2+8*((M_prop+M_motor)*(d^2));
            disp('Coaxial Quadcopter');
        case 3 %Simple Hexacopter
            Bf= [1 1 1 1 1 1; 
                0 -sqrt(3)/2*d -sqrt(3)/2*d 0 sqrt(3)/2*d sqrt(3)/2*d; 
                d 0.5*d -0.5*d -d -0.5*d 0.5*d; 
                j -j j -j j -j];
            maxinput = [1 1 1 1 1 1]'; %max thrust force (N)
            mininput = [0 0 0 0 0 0]';
            Ixx=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 4*((M_prop+M_motor)*(sqrt(3)*d/2)^2);
            Iyy=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 2*((M_prop+M_motor)*(d^2))+4*((M_prop+M_motor)*((d/2)^2));
            Izz=M_center*0.5^(2)/2+6*((M_prop+M_motor)*(d^2));
            disp('Simple Hexacopter');
        case 4 %Coaxial Hexacopter
            Bf= [1 1 1 1 1 1 1 1 1 1 1 1; 
                0 0 -sqrt(3)/2*d -sqrt(3)/2*d -sqrt(3)/2*d -sqrt(3)/2*d 0 0 sqrt(3)/2*d sqrt(3)/2*d sqrt(3)/2*d sqrt(3)/2*d;
                d  d  0.5*d 0.5*d -0.5*d -0.5*d -d -d -0.5*d -0.5*d  0.5*d 0.5*d;
                j -j  -j   j    j   -j  -j  j   j   -j   -j   j];
            maxinput = [1 1 1 1 1 1 1 1 1 1 1 1]'; %max thrust force (N)
            mininput = [0 0 0 0 0 0 0 0 0 0 0 0]';
            Ixx=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 8*((M_prop+M_motor)*(sqrt(3)*d/2)^2);
            Iyy=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 4*((M_prop+M_motor)*(d^2))+8*((M_prop+M_motor)*((d/2)^2));
            Izz=M_center*0.5^(2)/2+12*((M_prop+M_motor)*(d^2));
            disp('Coaxial Hexacopter');   
        case 5 %Simple Octocopter
            Bf= [1 1 1 1 1 1 1 1; 
                0  -d*sqrt(2)/2   -d   -d*sqrt(2)/2    0    d*sqrt(2)/2   d   d*sqrt(2)/2; 
                d   d*sqrt(2)/2    0   -d*sqrt(2)/2   -d   -d*sqrt(2)/2   0   d*sqrt(2)/2; 
                -j j -j j -j j -j j];
            maxinput = [1 1 1 1 1 1 1 1]'; %max thrust force (N)
            mininput = [0 0 0 0 0 0 0 0]';
            Ixx=(M_center*0.5^(2)/4)+(M_center*0.25^(2)/12) + 4*((M_prop+M_motor)*(sqrt(3)*d/2)^2)+2*((M_prop+M_motor)*(d^2));
            Iyy=Ixx;
            Izz=M_center*0.5^(2)/2+8*((M_prop+M_motor)*(d^2));
            disp('Simple Octorotor');   
    end    
    A = [0 0 0 0 1 0 0 0 ; 0 0 0 0 0 1 0 0 ; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1; 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 ;0 0 0 0 0 0 0 0;0 0 0 0 0 0 0 0];
    B_i = [0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0; 1/M_uav 0 0 0; 0 1/Ixx 0 0; 0 0 1/Iyy 0; 0 0 0 1/Izz];
    B=B_i*Bf;

    % Discretization-----------------------------------------------------------

    dT= T/N; % time step
    %J=B_i*Grav*dT;

    G = expm(A*dT);

    syms tau
    H = expm(A*dT)*double(int(expm(-A*tau),tau,0,dT))*B;
    % syms lambda
    % H = int(exp(-A*lambda),lambda,0,dT)*B;

    % i=0 iteration
    F = G^(N-1)*H;
    %Q= G^(N-1)*J;

    for i=1:(N-1)
            F = [F G^(N-1-i)*H];
    end

    % Failure injection--------------------------------------------------------
    rho=zeros(1,size(mininput,1));
    for j=1:size(mininput,1)

        disp(['Failure of Rotor ',num2str(j),' out of ',num2str(size(mininput,1))])
        if j==1
        else
            maxinput(j-1,1)=1;
        end
        maxinput(j,1)=0;
        a = mininput+pinv(Bf)*Grav; % min vector input (was a)
        b = fmax*maxinput+pinv(Bf)*Grav; % max vector input (was b)

        average = (a+b)/2; % Linear effector contribution magnitude vector
        delta =(b-a)/2;
        average_0 = average;
        delta_0 = delta;
        
        for i=1:(N-1)
                average = [average ; average_0];
                delta = [delta ; delta_0];
        end

        % x_init = -G^(-N)*F*u;
        % x_init = eval(x_init)

        %disp('Discretization complete');
        %disp('Compute DOC...');

    % DOC----------------------------------------------------------------------
        K = -inv(G^N)*F;
        %Uc=delta_0*ones([size(K,2),1]);
        xp=K*average; %Should be like Q, something isnt working
        %Uc(1)=center;

        % Define combinations of effectors of hyperplane segments
        [n_A m_A] = size(A); % Dimension of A (n_A = nb of state variables)
        [n_u m_u] = size(average_0); % Dimension of u (n_u = nb of effectors)
        M_uav=1:N*n_u;
        % S1=nchoosek(M,(N*n_u-(n_A-1))); % Combinations of N*n_u-(n_A-1) effector actions out-of N*n_u
        S1=nchoosek(M_uav,n_A-1); % Combinations of (n_A-1) effector actions out-of N*n_u
        % Compute DOC for each hyperplane segment
        [n_S1 m_S1] = size(S1);
        %d=zeros(1, n_S1);
        %L=zeros(1, n_S1);
        dL=zeros(1, n_S1);

        for i=1:n_S1
            %disp(['Failure of Rotor ',num2str(j),' out of ',num2str(size(mininput,1)),' Iteration ',num2str(i),' over ',num2str(n_S1)])
            % Define vector orth. to hypersegment col-space
            choose=S1(i,:); % S1 matrice of each combination of (n_A-1) effector actions
            K1=K(:,choose); % K1 matrice (hypersegment row-space)
            K2=K;
            K2(:,choose)=[];
            %u2{i}=average;
            u2=delta;
            u2(choose)=[];
            xi=null(K1'); % xi vector (orth. to hypersegment col-space)
            xi=xi(:,1); % Take first colum in case of a multiple column solution
            xi=xi/norm(xi); % Normalized xi vector

            % Compute distances
            %d(i)=max((xi{i}'*K2{i})*u2{i},[],"all"); %trials according to chinese paper, issues with size, not actually distances?
            %d(i)=max((sign(xi{i}'*K2{i}))'*((xi{i}'*K2{i})*u2{i}),[],"all"); %Is max necessary? u2 vector or variable?
            %d=(sign(xi{i}'*K2{i}))'*((xi{i}'*K2{i})*delta);
            d=abs(xi'*K2)*u2;
            L=abs(xi'*(xp));
            dL(i)=d-L; %issues with size
            %sign(dL{1,i}); %dL currently always negative
            % Sum of projected contribution magnitude of the effector actions 
            % out-of hyperplane segment.
        end

        rho(j) = (min(dL,[],"all"));

    end

    disp(['Minimal Controllability is ', num2str(min(rho,[],"all"))]);
    disp(['Below are all Degree of Controllability values']);
    rho
    DOC=min(rho,[],"all");
    toc
end

