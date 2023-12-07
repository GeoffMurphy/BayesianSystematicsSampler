''' Functions for adding/removing systematics to visibilities.
Most only differ by which systematics are added (i.e. 
for the autocorrelation and cross-correlation cases). 
Also differs by outputs: either a theano tensor object for
pymc3, or normal arrays for everything else, the latter being
faster to work with.'''

import numpy as np
import theano.tensor as T

def Auto_Systematics(refl_amp, refl_dly ,refl_phs,
                cc_amps, cc_dlys, cc_phs,
                subreflAmps, subreflDlys, subreflPhses,
                FGAmps, FG_Evecs, EORFAmps, EOR_FModes,
                freqs, NRefls, cc_Ncopies, NSubrefls, N_FAmps, N_FGAmps):
    
    CorrVisR = 0
    CorrVisIm = 0
    
    #Create the FG+EoR visibility from the eigenvectors and Fourier modes and their associated amplitudes
    
    for nn in range(0,N_FGAmps):
        CorrVisR += FG_Evecs[0][nn]*FGAmps[nn]
        CorrVisIm += FG_Evecs[1][nn]*FGAmps[nn]
    
    
    for uu in range(0,N_FAmps):
        CorrVisR += EOR_FModes[0][uu]*EORFAmps[uu]
        CorrVisIm += EOR_FModes[1][uu]*EORFAmps[uu]

    #Take the real component as the autocorrelation visibility    
    autovis = CorrVisR
    
    #Cross-coupling
    for qq in range(0,cc_Ncopies):

        CorrVisR = CorrVisR + autovis * cc_amps[qq] * np.cos(2*np.pi*freqs*cc_dlys[qq]+cc_phs[qq]) + autovis * cc_amps[qq] * np.cos(2*np.pi*freqs*-cc_dlys[qq]+cc_phs[qq])
        CorrVisIm = CorrVisIm + autovis * cc_amps[qq] * np.sin(2*np.pi*freqs*cc_dlys[qq]+cc_phs[qq]) + autovis * cc_amps[qq] * np.sin(2*np.pi*freqs*-cc_dlys[qq]+cc_phs[qq])
    
    #Subreflections
    for rr in range(0,NSubrefls):
        
        gi_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        gi_Im = subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        
        gi_conj_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        gi_conj_Im = -subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])

        
        gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
        gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R
        
        CorrVisR = CorrVisR * gain_R - CorrVisIm * gain_Im
        CorrVisIm = CorrVisR * gain_Im + CorrVisIm * gain_R
      
    #Reflections
    for bb in range(NRefls):

        gi_R = 1 + refl_amp[bb] * np.cos(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])
        gi_Im = refl_amp[bb] * np.sin(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])


        gi_conj_R = 1 + refl_amp[bb] * np.cos(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])
        gi_conj_Im = -refl_amp[bb] * np.sin(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])


        gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
        gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R

        CorrVisR = CorrVisR * gain_R - CorrVisIm * gain_Im
        CorrVisIm = CorrVisR * gain_Im + CorrVisIm * gain_R

    return T.stack([CorrVisR,CorrVisIm],axis=1)

'''---------------------------------------------------------------------------------------------'''

''' Same function as above, but returns array, as opposed to a tensor object (which is 
    slower to work with)'''

def Auto_Systematics_No_T(refl_amp, refl_dly ,refl_phs,
                cc_amps, cc_dlys, cc_phs,
                subreflAmps, subreflDlys, subreflPhses,
                FGAmps, FG_Evecs, EORFAmps, EOR_FModes,
                freqs, NRefls, cc_Ncopies, NSubrefls, N_FAmps, N_FGAmps):
    
    CorrVisR = 0
    CorrVisIm = 0
    
    #Create the FG+EoR visibility from the eigenvectors and Fourier modes and their associated amplitudes
    
    for nn in range(0,N_FGAmps):
        CorrVisR += FG_Evecs[0][nn]*FGAmps[nn]
        CorrVisIm += FG_Evecs[1][nn]*FGAmps[nn]
    
    
    for uu in range(0,N_FAmps):
        CorrVisR += EOR_FModes[0][uu]*EORFAmps[uu]
        CorrVisIm += EOR_FModes[1][uu]*EORFAmps[uu]

    #Take the real component as the autocorrelation visibility    
    autovis = CorrVisR
    
    #Cross-coupling
    for qq in range(0,cc_Ncopies):

        CorrVisR = CorrVisR + autovis * cc_amps[qq] * np.cos(2*np.pi*freqs*cc_dlys[qq]+cc_phs[qq]) + autovis * cc_amps[qq] * np.cos(2*np.pi*freqs*-cc_dlys[qq]+cc_phs[qq])
        CorrVisIm = CorrVisIm + autovis * cc_amps[qq] * np.sin(2*np.pi*freqs*cc_dlys[qq]+cc_phs[qq]) + autovis * cc_amps[qq] * np.sin(2*np.pi*freqs*-cc_dlys[qq]+cc_phs[qq])
    
    #Subreflections
    for rr in range(0,NSubrefls):
        
        gi_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        gi_Im = subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        
        gi_conj_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
        gi_conj_Im = -subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])

        
        gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
        gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R
        
        CorrVisR = CorrVisR * gain_R - CorrVisIm * gain_Im
        CorrVisIm = CorrVisR * gain_Im + CorrVisIm * gain_R
      
    #Reflections
    for bb in range(NRefls):

        gi_R = 1 + refl_amp[bb] * np.cos(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])
        gi_Im = refl_amp[bb] * np.sin(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])


        gi_conj_R = 1 + refl_amp[bb] * np.cos(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])
        gi_conj_Im = -refl_amp[bb] * np.sin(2 * np.pi * freqs * refl_dly[bb] + refl_phs[bb])


        gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
        gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R

        CorrVisR = CorrVisR * gain_R - CorrVisIm * gain_Im
        CorrVisIm = CorrVisR * gain_Im + CorrVisIm * gain_R

    return np.column_stack([CorrVisR,CorrVisIm])

'''---------------------------------------------------------------------------------------------'''

''' Simply invert the systematics operations to remove the systematics'''

def Auto_Subtract(Obs,refl_amp,refl_dly,refl_phs,
                cc_amps, cc_dlys, cc_phs,
             subreflAmps, subreflDlys, subreflPhses,
             FGAmps,FG_Evecs,EORFAmps,EOR_FModes,
                freqs, NRefls, cc_Ncopies, NSubrefls, N_FAmps, N_FGAmps):
    
    #refl_amp, refl_dly, refl_phs = theta
    
        CorrVisR = 0
        CorrVisIm = 0

        CorrVisR_FG = 0
        CorrVisIm_FG = 0 #Only use FG when adding systematics, else copies of EOR is added as well

        for nn in range(0,N_FGAmps):
            CorrVisR += FG_Evecs[0][nn]*FGAmps[nn]
            CorrVisIm += FG_Evecs[1][nn]*FGAmps[nn]


        for uu in range(0,N_FAmps):
            CorrVisR += EOR_FModes[0][uu]*EORFAmps[uu]
            CorrVisIm += EOR_FModes[1][uu]*EORFAmps[uu]

        autovisF = CorrVisR

        ObsR = Obs[:,0]
        ObsIm = Obs[:,1]

        '''Subreflections'''

        for rr in range(0,NSubrefls):

            gi_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
            gi_Im = subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])

            gi_conj_R = 1 + subreflAmps[rr] * np.cos(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])
            gi_conj_Im = -subreflAmps[rr] * np.sin(2 * np.pi * freqs * subreflDlys[rr] + subreflPhses[rr])

            gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
            gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R

            ObsR_temp = ObsR
            ObsIm_temp = ObsIm

            ObsR = (ObsR_temp*gain_R + ObsIm_temp*gain_Im)/(gain_R**2 + gain_Im**2)
            ObsIm = (ObsIm_temp*gain_R - ObsR_temp*gain_Im)/(gain_R**2 + gain_Im**2)

        '''Cable Reflection'''
        for zz in range(NRefls):
            gi_R = 1 + refl_amp[zz] * np.cos(2 * np.pi * freqs * refl_dly[zz] + refl_phs[zz])
            gi_Im = refl_amp[zz] * np.sin(2 * np.pi * freqs * refl_dly[zz] + refl_phs[zz])


            gi_conj_R = 1 + refl_amp[zz] * np.cos(2 * np.pi * freqs * refl_dly[zz] + refl_phs[zz])
            gi_conj_Im = -refl_amp[zz] * np.sin(2 * np.pi * freqs * refl_dly[zz] + refl_phs[zz])

            gain_R = gi_R*gi_conj_R - gi_Im*gi_conj_Im
            gain_Im = gi_R*gi_conj_Im + gi_Im*gi_conj_R

            ObsR_temp = ObsR
            ObsIm_temp = ObsIm

            ObsR = (ObsR_temp*gain_R + ObsIm_temp*gain_Im)/(gain_R**2 + gain_Im**2)
            ObsIm = (ObsIm_temp*gain_R - ObsR_temp*gain_Im)/(gain_R**2 + gain_Im**2)

        '''Cross-coupling'''


        amps = cc_amps
        dlys = cc_dlys
        phs = cc_phs


        for qq in range(0,cc_Ncopies):


            CorrVisR = CorrVisR + autovisF * amps[qq] * np.cos(2*np.pi*freqs*dlys[qq]+phs[qq]) + autovisF * amps[qq] * np.cos(2*np.pi*freqs*-dlys[qq]+phs[qq])
            CorrVisIm = CorrVisIm + autovisF * amps[qq] * np.sin(2*np.pi*freqs*dlys[qq]+phs[qq]) + autovisF * amps[qq] * np.sin(2*np.pi*freqs*-dlys[qq]+phs[qq])

            ObsR = ObsR - autovisF * amps[qq] * np.cos(2*np.pi*freqs*dlys[qq]+phs[qq]) - autovisF * amps[qq] * np.cos(2*np.pi*freqs*-dlys[qq]+phs[qq])
            ObsIm = ObsIm - autovisF * amps[qq] * np.sin(2*np.pi*freqs*dlys[qq]+phs[qq]) - autovisF * amps[qq] * np.sin(2*np.pi*freqs*-dlys[qq]+phs[qq])

        return np.column_stack([ObsR,ObsIm])
