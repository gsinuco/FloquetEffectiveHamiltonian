clear all;
clf;

%*****************************
% Settings
%*****************************

% Physcial Constants
au    = 1.6605389e-27; %
mCs   = 133*au;             %Atomic mass of cesium; 
hbar  = 1.0545718e-34;      %Plancks constant

% dipole traps
wr = 2*pi*50;
wz = 2*pi*1;               % trap frequencies

% harmonic oscillator width
Lr  = sqrt(hbar/(mCs*wr));     
Lz  = sqrt(hbar/(mCs*wz));    

% scattering length
a0    = 0.52917721092e-10;  % Bohr Radius
aScat = 200*a0;              % scattering length
g1D   = 2*hbar*hbar*aScat/(mCs*Lr^2); 


%*****************************
% Spatial grid
%*****************************
N  = 10000;
M  = 300; 
Nz = 2*M+1;
dz = double(5e-7); 
zV = (-M:1:M)*dz; 

%*****************************
% Momentum grid
%*****************************
dk = pi/(M*dz); 
kV =(-M:1:M)*dk; 

%*****************************
% Potential
%*****************************
V = 0.5*mCs*wz^2*zV.^2/hbar; 

%*****************************
% Wavefunction
%*****************************
psi_0 = sqrt(N/Lz)*(1/pi)^(1/4)*exp(-zV.^2/(10*Lz^2)); 

figure(1)
yyaxis left
plot(zV*1E6, abs(psi_0).^2, 'b' )
title('Start density');
ylabel('Density (arb.)');
yyaxis right
plot(zV*1E6, V, 'r' );
xlabel('position (um)');
ylabel('Potential (arb.)');
dz = ylim();
ylim(dz .*[1 1.2]); 


%*****************************
% Time
%*****************************
dt  = 1E-7;   % time steps
Nt  = 2000000; % number of steps                


%*****************************
% Loop over time
%*****************************
Nframe = 5000; %Data saved every Nframe steps

tV   = 0; 
psi  = psi_0; 

denseMatrix = zeros(Nt/Nframe,Nz); %Initialization
timeVect  = zeros(1,Nt);

for itime=1:Nt %Time-stepping with split-step Fourier method

    % get current potential
    psi   = psi.*exp(-0.5*1i*dt*(V+(g1D/hbar).*abs(psi).^2));
    psi_k = fftshift(fft(psi)/Nz);
    psi_k = psi_k.*exp(-0.5*dt*1i*(hbar/mCs)*kV.^2);
    psi   = ifft(ifftshift(psi_k))*Nz;
    psi   = psi.*exp(-0.5*1i*dt*(V+(g1D/hbar).*abs(psi).^2));
    if mod(itime,500) == 0  
        denseMatrix(itime,:) = abs(psi.^2); 
        timeVect(itime) = dt*(itime-1);
    end
        
    if mod(itime,Nframe) == 0
       % plot result
        figure(2)
        yyaxis left
        plot(zV*1E6, denseMatrix(end,:), 'b' )
        ylabel('Density (arb.)');
        yyaxis right
        plot(zV*1E6, V, 'r' );
        xlabel('position (um)');
        ylabel('Potential (arb.)');
        dz = ylim();
        ylim(dz .*[1 1.2]); 
        fprintf('Propagete: %.2f.\n',itime);
        title(sprintf('t=%f\n',timeVect(itime)));
        drawnow();
    end
end

return;

subplot(1,3,1); %Plot potential
plot(x,V,'k'); 
xlabel('x (m)'); 
ylabel('V (J/hbar)');

subplot(1,3,2); %Plot initial and final density
plot(x,abs(psi_0).^2,'k',x,abs(psi).^2,'b');

legend('\psi(x,0)','\psi(x,T)');
xlabel('x (m)');
ylabel('|\psi|ˆ2 (mˆ{-1})');

subplot(1,3,3); % Plot spacetime evolution as pcolor plot
dt_large=dt*double(Nt/Nframe);
pcolor(x,dt_large*(1:1:Nframe),spacetime); 
shading interp;
xlabel('x (m)'); 
ylabel('t (s)');

 