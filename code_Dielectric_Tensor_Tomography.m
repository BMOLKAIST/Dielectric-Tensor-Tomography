% code by Seungwoo Shin
% from the paper : 'Tomographic measurement of dielectric tensors at
% optical frequency', Nature Materials, 21, 317–324 (2022)
% The test data corresponding to the Fig.3c can be downloaded here: 
% https://drive.google.com/file/d/1eg7sAlkmvuZJTf39--So8aLkd4la9b5Q/view?usp=sharing
clear; clc; close all
load('data.mat')
%% Dielectric Tensor Tomography
k0_zL=real(sqrt((n_m*k0)^2-(k0_xL).^2-(k0_yL).^2));
k0_zR=real(sqrt((n_m*k0)^2-(k0_xR).^2-(k0_yR).^2));
k0_z_Rst=real(sqrt((n_m*k0)^2-(k0_x_Rst).^2-(k0_y_Rst).^2));
dKx_st = (-k0_x_Rst/kpx) - (-k0_xL/kpx);
dKy_st = (-k0_y_Rst/kpx) - (-k0_yL/kpx);
dKz_st = (-k0_z_Rst/kpx) - (-k0_zL/kpx);
dR_st = illu_RCPst - illu_RCP;
px3=px2*ZP/ZP2; % The lateral voxel size
px4=px2*ZP/ZP3; % The axial voxel size
er = round(2*NA/lambda/kpx);
Emask = ~mk_ellipse(er,er,er,ZP2,ZP2,ZP3);% Far-field measurement

FFxx=gpuArray(single(zeros(ZP2,ZP2,ZP3))); % scattering potential F
FFxy=gpuArray(single(zeros(ZP2,ZP2,ZP3)));
FFxz=gpuArray(single(zeros(ZP2,ZP2,ZP3)));
FFyy=gpuArray(single(zeros(ZP2,ZP2,ZP3)));
FFyz=gpuArray(single(zeros(ZP2,ZP2,ZP3)));
FFzz=gpuArray(single(zeros(ZP2,ZP2,ZP3)));
Count=gpuArray(single(zeros(ZP2,ZP2,ZP3)));

xr=round(ZP*px2*NA/lambda);
[kx ky]=meshgrid(kpx*(-floor(ZP/2):floor(ZP/2)-1),kpx*(-floor(ZP/2):floor(ZP/2)-1)); % unit: (um^-1) % set kx-ky coordinate
kz=real(sqrt((n_m*k0)^2-kx.^2-ky.^2)); % unit: (um^-1) % Generating coordinates on the surface of Ewald Sphere
[xx,yy,zz] = meshgrid((-ZP2/2:ZP2/2-1),(-ZP2/2:ZP2/2-1),(-ZP3/2:ZP3/2-1));
[xx2,yy2] = meshgrid(-ZP/2:ZP/2-1);
thr = 5*pi/180;
nosingul = 0;

for kk= 1 : thetaSize
    if (theta(kk)>thr/2)&&(abs(phi(kk))>thr)&&(abs(phi(kk)-pi/2)>thr)&&(abs(phi(kk)-pi)>thr)...
            &&(abs(phi(kk)-pi/2*3)>thr)&&(abs(phi(kk)-2*pi)>thr)&&(abs(phi(kk)+pi/2)>thr)...
            &&(abs(phi(kk)+pi)>thr)&&(abs(phi(kk)+pi/2*3)>thr) % to avoid singularities
        figure(10), plot(k0_xL(kk),k0_yL(kk),'ob'),hold on,axis image
        xlim([min(k0_xL),max(k0_xL)]),ylim([min(k0_yL),max(k0_yL)]),grid on
        title('(k_{0x}, k_{0y}) except for the singularities')
        Kx=kx-k0_xL(kk);Ky=ky-k0_yL(kk);Kz=kz-k0_zL(kk);
        xind=find((kz>0).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*(Kx>(kpx*(-floor(ZP2/2))))...
            .*(Ky>(kpx*(-floor(ZP2/2))))...
            .*(Kz>(kpx*(-floor(ZP3/2))))...
            .*(Kx<(kpx*(floor(ZP2/2)-1)))...
            .*(Ky<(kpx*(floor(ZP2/2)-1)))...
            .*(Kz<(kpx*(floor(ZP3/2)-1))));
        Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind);
        Kx=round(Kx/kpx+ZP2/2+1); Ky=round(Ky/kpx+ZP2/2+1); Kz=round(Kz/kpx+ZP3/2+1);
        Kzp=(Kz-1)*ZP2^2+(Kx-1)*ZP2+Ky;
        Kx2=kx-k0_x_Rst(kk);Ky2=ky-k0_y_Rst(kk);Kz2=kz-k0_z_Rst(kk);
        xind=find((kz>0).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*(Kx2>(kpx*(-floor(ZP2/2))))...
            .*(Ky2>(kpx*(-floor(ZP2/2))))...
            .*(Kz2>(kpx*(-floor(ZP3/2))))...
            .*(Kx2<(kpx*(floor(ZP2/2)-1)))...
            .*(Ky2<(kpx*(floor(ZP2/2)-1)))...
            .*(Kz2<(kpx*(floor(ZP3/2)-1))));
        Kx2=Kx2(xind); Ky2=Ky2(xind); Kz2=Kz2(xind);
        Kx2=round(Kx2/kpx+ZP2/2+1); Ky2=round(Ky2/kpx+ZP2/2+1); Kz2=round(Kz2/kpx+ZP3/2+1);
        Kzp2=(Kz2-1)*ZP2^2+(Kx2-1)*ZP2+Ky2;
        PsixRCP = gpuArray(squeeze(log(eexAmpRCP(:,:,kk))+1i*eexPhRCP(:,:,kk)));
        PsiyRCP = gpuArray(squeeze(log(eeyAmpRCP(:,:,kk))+1i*eeyPhRCP(:,:,kk)));
        PsizRCP = gpuArray(squeeze(log(eezAmpRCP(:,:,kk))+1i*eezPhRCP(:,:,kk)));
        PsixRCPst = gpuArray(squeeze(log(eexAmpRCPst(:,:,kk))+1i*eexPhRCPst(:,:,kk)));
        PsiyRCPst = gpuArray(squeeze(log(eeyAmpRCPst(:,:,kk))+1i*eeyPhRCPst(:,:,kk)));
        PsizRCPst = gpuArray(squeeze(log(eezAmpRCPst(:,:,kk))+1i*eezPhRCPst(:,:,kk)));
        PsixLCP = gpuArray(squeeze(log(eexAmpLCP(:,:,kk))+1i*eexPhLCP(:,:,kk)));
        PsiyLCP = gpuArray(squeeze(log(eeyAmpLCP(:,:,kk))+1i*eeyPhLCP(:,:,kk)));
        PsizLCP = gpuArray(squeeze(log(eezAmpLCP(:,:,kk))+1i*eezPhLCP(:,:,kk)));
        l1 = illu_LCP(1,kk); l2 = illu_LCP(2,kk); l3 = illu_LCP(3,kk);
        r1 = illu_RCP(1,kk); r2 = illu_RCP(2,kk); r3 = illu_RCP(3,kk);
        tmp = gpuArray(single(zeros(ZP2,ZP2,ZP3)));
        Mx = gpuArray(r1*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsixRCP.*exp(1i*2*pi*(k0_xR(kk)/kpx*xx2/ZP+k0_yR(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = Mx(xind);
        clear PsixRCP Mx % to save the gpu memory
        Mxr = fftshift(ifftn(ifftshift(tmp)));
        My = gpuArray(r2*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsiyRCP.*exp(1i*2*pi*(k0_xR(kk)/kpx*xx2/ZP+k0_yR(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = My(xind);
        clear PsiyRCP My
        Myr = fftshift(ifftn(ifftshift(tmp)));
        Mz = gpuArray(r3*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsizRCP.*exp(1i*2*pi*(k0_xR(kk)/kpx*xx2/ZP+k0_yR(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = Mz(xind);
        clear PsizRCP Mz
        Mzr = fftshift(ifftn(ifftshift(tmp)));
        Nx = gpuArray(l1*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsixLCP.*exp(1i*2*pi*(k0_xL(kk)/kpx*xx2/ZP+k0_yL(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = Nx(xind);
        clear PsixLCP Nx
        Nxr = fftshift(ifftn(ifftshift(tmp)));
        Ny = gpuArray(l2*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsiyLCP.*exp(1i*2*pi*(k0_xL(kk)/kpx*xx2/ZP+k0_yL(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = Ny(xind);
        clear PsiyLCP Ny
        Nyr = fftshift(ifftn(ifftshift(tmp)));
        Nz = gpuArray(l3*(kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsizLCP.*exp(1i*2*pi*(k0_xL(kk)/kpx*xx2/ZP+k0_yL(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp) = Nz(xind);
        clear PsizLCP Nz
        Nzr = fftshift(ifftn(ifftshift(tmp)));
        tmp = gpuArray(single(zeros(ZP2,ZP2,ZP3)));
        Qz = gpuArray((kz/1i).*~mk_ellipse(xr,xr,ZP,ZP)...
            .*fftshift(fft2(ifftshift(PsizRCPst.*exp(1i*2*pi*(k0_x_Rst(kk)/kpx*xx2/ZP+k0_y_Rst(kk)/kpx*yy2/ZP)))))*(px2)^2);
        tmp(Kzp2) = Qz(xind);
        clear PsizRCPst Qz
        Qzr = fftshift(ifftn(ifftshift(tmp)));
        dpx = dR_st(1,kk); dpy = dR_st(2,kk); dpz = dR_st(3,kk);
        inv_ramp = gpuArray(1./(1-(1i*2*pi*( dKx_st(kk)*xx/ZP2+dKy_st(kk)*yy/ZP2+dKz_st(kk)*zz/ZP2))));
        ZZ =(Mzr*dpx*l2 - Mzr*dpy*l1 - Nzr*dpx*r2 + Nzr*dpy*r1 + Qzr*dpz.*inv_ramp*l1*r2 - Qzr*dpz.*inv_ramp*l2*r1)...
            ./(dpx*l2*r3 - dpx*l3*r2 - dpy*l1*r3 + dpy*l3*r1 + dpz*l1*r2 - dpz*l2*r1);
        FFzz = FFzz + ZZ;
        tmp =(l1*(Mzr - ZZ*r3) - Nzr*r1 + ZZ*l3*r1)/(l1*r2 - l2*r1);
        FFyz = FFyz + tmp;
        tmp =-(Mzr*l2 - Nzr*r2 - ZZ*l2*r3 + ZZ*l3*r2)/(l1*r2 - l2*r1);
        FFxz = FFxz + tmp;
        tmp =-(l2*(Myr - (r3*(l1*(Mzr - ZZ*r3) - Nzr*r1 + ZZ*l3*r1))/(l1*r2 - l2*r1))*(l1*r2 - l2*r1) + l3*r2*(l1*(Mzr - ZZ*r3) - Nzr*r1 + ZZ*l3*r1) - Nyr*r2*(l1*r2 ...
            - l2*r1))/(l1*r2*(l1*r2 - l2*r1) - l2*r1*(l1*r2 - l2*r1));
        FFxy = FFxy + tmp;
        tmp =(ZZ*l1^2*r3^2 + ZZ*l3^2*r1^2 + Myr*l1^2*r2 - Mzr*l1^2*r3 + Nyr*l2*r1^2 - Nzr*l3*r1^2 - Myr*l1*l2*r1 + Mzr*l1*l3*r1 - Nyr*l1*r1*r2 + Nzr*l1*r1*r3 ...
            - 2*ZZ*l1*l3*r1*r3)/(l1*r2 - l2*r1)^2;
        FFyy = FFyy + tmp;
        tmp =((r1*(Mxr*l1*r2 - Mxr*l2*r1 + Mzr*l2*r3 - Nzr*r2*r3 - ZZ*l2*r3^2 + ZZ*l3*r2*r3))/(l1*r2 - l2*r1) + (r1*r2*(l2*(Myr*l1*r2 - Myr*l2*r1 - Mzr*l1*r3 ...
            + Nzr*r1*r3 + ZZ*l1*r3^2 - ZZ*l3*r1*r3) + l3*r2*(Mzr*l1 - Nzr*r1 - ZZ*l1*r3 + ZZ*l3*r1) - Nyr*r2*(l1*r2 - l2*r1)))/(l1*r2 - l2*r1)^2)/r1^2;
        FFxx = FFxx + tmp;
        clear inv_ramp  Mxr  Myr  Mzr  Nxr  Nyr  Nzr Qzr ZZ
        Count(Kzp)=Count(Kzp)+1;
        nosingul = nosingul +1;
        disp(['Reconstruction progress: ',num2str(kk),' / ',num2str(thetaSize)])
    end
end
mask = single(gather(Count)>0);
clear exxAmpLCP eexAmpRCP eexAmpRCPst eexPhLCP  eexPhRCP  eexPhRCPst  eeyAmpLCP  eeyAmpRCP  eeyAmpRCPst  eeyPhLCP  eeyPhRCP  eeyPhRCPst  eezAmpLCP  eezAmpRCP  eezAmpRCPst  eezPhLCP  eezPhRCP eezPhRCPst

FFxx = fftshift(fftn(ifftshift(FFxx))).*(Count>0);
FFxx(Count>0)=FFxx(Count>0)./Count(Count>0)/(px4*px3^2); % should be (um^-2)*(px*py*pz), so (px*py*pz/um^3) should be multiplied.
FFxy = fftshift(fftn(ifftshift(FFxy))).*(Count>0);
FFxy(Count>0)=FFxy(Count>0)./Count(Count>0)/(px4*px3^2);
FFxz = fftshift(fftn(ifftshift(FFxz))).*(Count>0);
FFxz(Count>0)=FFxz(Count>0)./Count(Count>0)/(px4*px3^2);
FFyy = fftshift(fftn(ifftshift(FFyy))).*(Count>0);
FFyy(Count>0)=FFyy(Count>0)./Count(Count>0)/(px4*px3^2);
FFyz = fftshift(fftn(ifftshift(FFyz))).*(Count>0);
FFyz(Count>0)=FFyz(Count>0)./Count(Count>0)/(px4*px3^2);
FFzz = fftshift(fftn(ifftshift(FFzz))).*(Count>0);
FFzz(Count>0)=FFzz(Count>0)./Count(Count>0)/(px4*px3^2);
clear Count
Fxx = gather(fftshift(ifftn(ifftshift(FFxx))));
Fxy = gather(fftshift(ifftn(ifftshift(FFxy))));
Fxz = gather(fftshift(ifftn(ifftshift(FFxz))));
Fyy = gather(fftshift(ifftn(ifftshift(FFyy))));
Fyz = gather(fftshift(ifftn(ifftshift(FFyz))));
Fzz = gather(fftshift(ifftn(ifftshift(FFzz))));
clear FFxx FFxy FFxz FFyy FFyz FFzz

exx = real(n_m^2*(1+4*pi*Fxx/(2*pi*n_m*k0)^2));
eyy = real(n_m^2*(1+4*pi*Fyy/(2*pi*n_m*k0)^2));
ezz = real(n_m^2*(1+4*pi*Fzz/(2*pi*n_m*k0)^2));
exy = real(n_m^2*(4*pi*Fxy/(2*pi*n_m*k0)^2));
exz = real(n_m^2*(4*pi*Fxz/(2*pi*n_m*k0)^2));
eyz = real(n_m^2*(4*pi*Fyz/(2*pi*n_m*k0)^2));
%% Singular Value Decomposition
u = single(zeros(size(exx))); v = single(zeros(size(exx))); w = single(zeros(size(exx)));
e1 = single(zeros(size(exx))); e2 = single(zeros(size(exx))); e3 = single(zeros(size(exx)));
for rr = 1 : length(exx(:))
    tmp = real([exx(rr),exy(rr),exz(rr);
        exy(rr),eyy(rr),eyz(rr);
        exz(rr),eyz(rr),ezz(rr)]);
    [U,S,~] = svd(tmp);
    u(rr) = U(1,1);
    v(rr) = U(2,1);
    w(rr) = U(3,1);
    e1(rr) = S(1,1);
    e2(rr) = S(2,2);
    e3(rr) = S(3,3);
end
n1 = reshape(sqrt(e1),size(exx));
n2 = reshape(sqrt(e2),size(exx));
n3 = reshape(sqrt(e3),size(exx));
u = reshape(u,size(exx));
v = reshape(v,size(exx));
w = reshape(w,size(exx));
%% Visualization directors
[xx,yy,zz] = meshgrid(px3*(-ZP2/2:ZP2/2-1),px3*(-ZP2/2:ZP2/2-1),px4*(-ZP3/2:ZP3/2-1));
tmp = [256,265,99];
x = tmp(1);
y = tmp(2);
z = tmp(3);

ZP2 = 160;
ZP3 = round(ZP2*px3/px4/2)*2;

xroi = x-ZP2/2:x+ZP2/2-1;
yroi = y-ZP2/2:y+ZP2/2-1;
zroi = z-ZP3/2:z+ZP3/2-1;

bn = 0.012;
bb = single(n1/n_m>(1+ bn));

xx0 = xx(yroi,xroi,zroi);
yy0 = yy(yroi,xroi,zroi);
zz0 = zz(yroi,xroi,zroi);

u0 = u(yroi,xroi,zroi);
v0 = v(yroi,xroi,zroi);
w0 = w(yroi,xroi,zroi);
bb0 = bb(yroi,xroi,zroi);

vz = round(size(xx0,3)/2)+1;
vx = round(size(xx0,2)/2)+1;
vy = round(size(xx0,1)/2)+1;

tmpn1 = n1(yroi,xroi,zroi);
tmpu = bb0.*u0;
tmpv = bb0.*v0;
tmpw = bb0.*w0;

figure(12), 
subplot(221),imagesc(xx0(1,1:end,vz),yy0(1:end,1,vz),tmpn1(:,:,vz),[n_m+0.002,max(n1(:))]),axis image,colormap('hot')
hold on, % tmp = sqrt(tmpu(:,:,vz).^2+tmpv(:,:,vz).^2); S = double(max(tmp(:)));
quiver(xx0(:,:,vz),yy0(:,:,vz),tmpu(:,:,vz),tmpv(:,:,vz),'MaxHeadSize',0,'Color','black','LineWidth',0.5),axis image,
title('x axis'),ylabel('y axis','FontWeight','bold')

subplot(222),imagesc(squeeze(zz0(1,vx,[1:end])),squeeze(yy0([1:end],vx,1)),squeeze(tmpn1(:,vx,:)),[n_m+0.002,max(n1(:))]),axis image,colormap('hot')
hold on, tmp = sqrt(tmpv(:,vx,:).^2+tmpw(:,vx,:).^2); S = double(max(tmp(:)));
quiver(squeeze(zz0(:,vx,:)),squeeze(yy0(:,vx,:)),...
    squeeze(tmpw(:,vx,:)),squeeze(tmpv(:,vx,:)),S,'MaxHeadSize',0,'Color','black','LineWidth',0.5),
axis image,title('z axis'), 

subplot(223),imagesc(squeeze(xx0(vy,[1:end],1)),squeeze(zz0(vy,1,[1:end])),circshift(imrotate(squeeze(tmpn1(vy,:,:)),-90),[0,1]),[n_m+0.002,max(n1(:))]),axis image,colormap('hot')
hold on, tmp = sqrt(tmpu(vy,:,:).^2+tmpw(vy,:,:).^2); S = double(max(tmp(:)));
quiver(squeeze(xx0(vy,:,:)),squeeze(zz0(vy,:,:)),...
    squeeze(tmpu(vy,:,:)),squeeze(tmpw(vy,:,:)),S,'MaxHeadSize',0,'Color','black','LineWidth',0.5),
axis image, ylabel('z axis','FontWeight','bold')
