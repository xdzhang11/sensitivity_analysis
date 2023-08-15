
import os
import numpy as np

class SoilFile(object):

    def __init__(self, filename=None):
        # self.soil_sets = {}
        self.force_len = []
        self.x = []
        if filename is not None:
            with open (filename) as fid:
                lines = fid.readlines()
            self._parse_lines(lines)
            self.filename = filename
        self.fmt = '.2f'
        
        
    def _parse_lines(self,lines):
        """Read HAWC2 soil file (soil dat file).
        """
        nsets = 1
        n_springs = 3
        lptr = 3
        force_len = []
        x = []
        n_rows = int(lines[lptr].split()[0])
        n_defls = int(lines[lptr].split()[1])
        for n_spring in range(1,n_springs+1):
            lptr += 1
            x.append(np.asarray(lines[lptr].split()).astype(float))
            lptr += 1
            data = np.array([[float(v) for v in l.split()[:n_defls+1]] for l in lines[lptr:lptr + n_rows]])
            force_len.append(data)
            lptr += n_rows+3
        self.force_len = force_len
        self.x = x


    def __str__(self, comments=None):
        """This method will create a string that is formatted like a pc file
        with the data in this class.
        """
        heads = 'Linear soil model, only lateral stiffness'
        spring_types = ['lateral [m] and [kN/m] \n', 'axial [m] and [kN/m]\n', 'rotation_z [rad] and [kN/rad] \n']
        
        linefmt_x = '\t'.join(['{%i:%s}' % (i, self.fmt) for i in range(3)])
        linefmt = '\t'.join(['{%i:%s}' % (i, self.fmt) for i in range(4)])
        retval = str(heads) + '\n'
        for i, (tc, pc) in enumerate(zip(self.x, self.force_len)):
                nr = pc.shape[0]
                nc = pc.shape[1]-1
                retval += '#%i \n' % (i+1)
                retval += spring_types[i]
                retval += '%i %i \n' % (nr, nc)
                retval += '\t\t'+linefmt_x.format(*tc)+ '\n'
                for line in pc:
                    retval += linefmt.format(*line) + '\n'
                retval += '-----------------------------------------------------------------\n'
        return retval


    def set_spring(self, set_number, fl):
        self.force_len[set_number] = fl

   
    def save(self, filename):
        if not os.path.isdir(os.path.dirname(filename)):
            # fails if dirname is empty string
            if len(os.path.dirname(filename)) > 0:
                os.makedirs(os.path.dirname(filename))
        with open(filename, 'w') as fid:
            fid.write(str(self))
        self.filename = filename


if __name__ == "__main__":
    soilfile = SoilFile("data/DTU_10MW_RWT_Soil.dat")

    print (soilfile.CL(21,10)) # CL for thickness 21% and AOA=10deg
    #1.358
    print (soilfile.CD(21,10)) # CD for thickness 21% and AOA=10deg
    #0.0255
    print (soilfile.CM(21,10)) # CM for thickness 21% and AOA=10deg
    #-0.1103

