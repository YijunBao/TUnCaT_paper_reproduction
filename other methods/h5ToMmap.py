import glob
import sys
sys.path.insert(0, 'C:\\Other methods\\CaImAn') # The folder containing the caiman code
import caiman as cm

datadir = '..\\data\\*\\'
AllFiles = glob.glob(datadir+"*.h5") + glob.glob(datadir+"SNR video\\*.h5")

# LOAD MOVIE AND MEMORYMAP
for DataFile in AllFiles:
    fname_new = cm.save_memmap([DataFile],
                                base_name =DataFile.rsplit('\\', 1)[-1][0:-3]+'_memmap_', 
                                order='C')