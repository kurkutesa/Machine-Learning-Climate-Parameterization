% This function converts water vapor UNITS.
% Usage:
%         function [wout] = convert_h2o(p,t,winp,unitinp,unitout)
% Input:  p       - Pressure in hPa
%         t       - Temperature in K
%         winp    - Water Vapor concentration in
%         unitinp - Input units
%         unitout - Output units
%
%         unitinp,unitout, character
%                         'M' Mass Mixing Ratio (g/Kg)
%                         'V' Volume Mixing Ratio (ppv)
%                         'H' Relative Umidity (%)
%                         'S' Specific Humidity
%                         'D' Dew Point (K)
%                         'P' Partial Pressure (hPa)
%                         'N' Number Density (cm-3)
%                         'R' Mass Density (kg/m3)
%                         'C' Colummn (mm) Output Only.
%
% Guido Masiello, April 2005, September 2009, May 2017
% Guido Masiello, September 2017, Optimization for Dewpoint Output.
%                 In the current version the routine solves analytically
%                 Clausius - Clapeyron law avoiding iterative approach of
%                 matlab fsolve function (used in the past version).
%
function [wout] = cnvrth2o(p,t,winp,unitinp,unitout)
avogad=6.02214199e+23;    % Avogadro Number
alosmt=2.6867775e+19;     % Loschmidt Number
airmwt=28.964;            % Air Molecular weight (grams)
amwt1=18.015;             % H2O Molecular weigth (grams)
g=9.806;                  % gravity acceleration.
c1=18.9766;               %
c2=-14.9595;              % Clausius - Clapeyron Constants
c3=-2.43882;              %
p0=1013.25;t0=273.15;
r = airmwt/amwt1;
b = avogad/amwt1;

p=reshape(p,length(p),1);
t=reshape(t,length(t),1);
winp=reshape(winp,length(winp),1);
rhoair = alosmt*(p./p0).*t0.*t.^-1; % Air Density (molecules/cm3)
%
% Converts Input water vapor in number density.
%
switch unitinp
    % Input Mass Mixing Ratio
    case 'M'
        dennum=r*winp.*(1.e3.*ones(size(winp))+r.*winp).^(-1).*rhoair;
    % Input Volume Mixing Ratio
    case 'V'
        dennum = winp.*(ones(size(winp))+winp).^(-1).*rhoair;
    % Input Relative Umidity
    case 'H'
        a = t0*t.^-1;
        % saturation pressure
        densat = a.*b.*exp(c1*ones(size(a))+c2.*a+c3*a.^2)*1.0E-6 ;
        dennum = densat.*(winp./100.0);
    % Input Specific Humidity
    case 'S'
        dennum = r.*winp.*rhoair;
    % Input Dew Point
    case 'D'
        a = t0*winp.^-1;
        % saturation pressure
        densat = a.*b.*exp(c1*ones(size(a))+c2.*a+c3*a.^2)*1.0E-6 ;
        dennum = densat.*(winp./t);
        % Input Partial Pressure
    case 'P'
        a = t0*t.^-1;
        dennum = alosmt.*(winp./p0).*a;
    % Input Number Density
    case 'N'
        dennum=winp;
    % Input Mass Density
    case 'R'
        dennum = b.*winp.*1.0e-6;
    otherwise
        disp(['WARNING: Flag for Input Units ' unitinp ' Not defined'])
        disp(['Avaible Choises:'])
        disp(['M - Mass Mixing Ratio (g/Kg)']);
        disp(['V - Volume Mixing Ratio (ppv)']);
        disp(['H - Relative Umidity (%)']);
        disp(['S - Specific Humidity']);
        disp(['D - Dew Point (K)']);
        disp(['P - Partial Pressure (hPa)']);
        disp(['N - Number Density (cm-3)']);
        disp(['R - Mass Density (kg/m3)']);
end
%
% Converts number density to the output units
%
switch unitout
    % Output Mass Mixing Ratio
    case 'M'
        wout=dennum./(rhoair-dennum)./(r*1.e-3);
    % Output Volume Mixing Ratio
    case 'V'
        wout=dennum./(rhoair-dennum);
    % Output Relative Humidity
    case 'H'
        a = t0*t.^-1;
        densat = a.*b.*exp(c1*ones(size(a))+c2.*a+c3*a.^2)*1.0E-6 ;
        wout=100.*dennum./densat;
    % Output Specific Humidity
    case 'S'
        wout = dennum.*(r.*rhoair).^(-1);
    % Output Dew Point
    case 'D'
        wout=nan(size(dennum));
        for i=1:length(dennum)
%             [x,y,f] = fsolve(@(x) exp(c1+c2.*x+c3*x.^2)-(dennum(i)*1e6./b)*t(i)./t0,1,optimset('Display','off'));
%             if f~=1
%                 f
%             end
            x=0.5*(sqrt((c2/c3)^2-4*(c1/c3-log(1.e6*dennum(i)/b*t(i)./t0)/c3))-c2/c3);
            wout(i,1)=t0/x;
        end
    % Output Partial Pressure
    case 'P'
        a = t0*t.^-1;
        wout=p0.*dennum./a./alosmt;
    % Output Number Density
    case 'N'
        wout=dennum;
    % Output Mass density
    case 'R'
        wout=dennum./b*1.0e+6;
    % Output Colummnar content
    case 'C'
        % % Compute specific Umidity
        % q = dennum.*(r.*rhoair+(1-2.*r)*dennum).^(-1);
        % % Integrate over the colummn
        % wout=0.0;
        % for i=1:length(p)-1
        %     wout=wout+0.5/g*(q(i)+q(i+1)).*100.*(p(i)-p(i+1));
        % end
        % Compute mass Mixing Ratio
        mmr=dennum./(rhoair-dennum)./(r*1.e-3);
        % Integrate over the colummn
        wout=0.0;
        for i=1:length(p)-1
            wout=wout+0.5/g*(mmr(i)+mmr(i+1)).*0.1*(p(i)-p(i+1));
        end
    otherwise
        disp(['WARNING: Flag for Output Units ' unitout ' Not defined'])
        disp(['Avaible Choices:'])
        disp(['M - Mass Mixing Ratio (g/Kg)']);
        disp(['V - Volume Mixing Ratio (ppv)']);
        disp(['H - Relative Umidity (%)']);
        disp(['S - Specific Humidity']);
        disp(['D - Dew Point (K)']);
        disp(['P - Partial Pressure (hPa)']);
        disp(['N - Number Density (cm-3)']);
        disp(['R - Mass Density (kg/m3)']);
        disp(['C - Colummnar content (mm)']);
end
if (exist('wout')==0)
    wout=NaN
end
return
