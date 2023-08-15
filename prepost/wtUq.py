from wetb.hawc2.Hawc2io import ReadHawc2 
from wetb.hawc2 import HTCFile, StFile
from pc_file import PCFile
from soil_file import SoilFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.stats import qmc, norm
from scipy.io import savemat, loadmat
from scipy.interpolate import interp1d
import os

class wtUq(object): # wind turbine Uncertainty quantificaiton

    def __init__(self, n, nSeed):
        self.n = n
        self.nSeed = nSeed
        self.htc = HTCFile(os.path.join('htc', 'dtu10mw_rwt.htc'))

    def _value(self):
        return 1

    def generate_seed(self, fName):
        n = self.n
        nSeed = self.nSeed
        seed = np.random.randint(1,1e6,size=(n,nSeed))
        f = h5py.File(fName, "w")
        f.create_dataset('seed', data=seed)
        f.close()

    
    def generate_qms(self, varList, fName):
        n = self.n
        d = len(varList)
        sampler = qmc.Halton(d)
        halton = sampler.random(n)
        qms = pd.DataFrame(data = halton, columns=varList)
        qms.to_hdf(fName, key = 'qms', mode = 'w')

    
    def generate_wsp(self, qms, seed):
        # %% Wind speed
        # mean wind speed
        x = qms.loc[:,"wsp"]
        n = self.n
        nSeed = self.nSeed
        u = 3+x*(27-3)
        htc = self.htc
        # htc = HTCFile(r".\htc\dtu10mw_rwt.htc")
        for j in range(n):
            htc.wind.wsp = u[j]
            htc.wind.tint = 0.2
            htc.wind.wind_ramp_factor[2] = 7.8/u[j]
            turb_dx = u[j]*600/8192
            htc.wind.mann.box_dim_u[1] = turb_dx
            turb_dy = 178.3*1.15/32
            htc.wind.mann.box_dim_v[1] = turb_dy
            turb_dz = turb_dy
            htc.wind.mann.box_dim_w[1] = turb_dz
            for k in range(nSeed):
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()
                
    def generate_ti(self, qms, seed):
        #%% sigma
        x = qms.loc[:,"ti"]
        n = self.n
        nSeed = self.nSeed
        htc = self.htc
        for j in range(n):
            u = float(np.array(htc.wind.wsp.values))
            sigma_min = np.max([0, 0.1 * (u - 20)])
            sigma_max = 0.18 * (6.8 + 0.75 * u)
            sigma = sigma_min + (sigma_max - sigma_min) * x[j]
            ti = sigma/u
            htc.wind.tint = ti
            htc.wind.wind_ramp_factor[2] = 7.8/u
            turb_dx = u*600/8192
            htc.wind.mann.box_dim_u[1] = turb_dx
            turb_dy = 178.3*1.15/32
            htc.wind.mann.box_dim_v[1] = turb_dy
            turb_dz = turb_dy
            htc.wind.mann.box_dim_w[1] = turb_dz
            for k in range(nSeed):
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()


    def generate_cl(self, qms, seed):
        # Lift coefficient Cl
        x = qms.loc[:,"cl"]
        n = self.n
        nSeed = self.nSeed
        htc = self.htc
        for j in range(n):
            pcfile = PCFile(os.path.join('data', 'DTU_10MW_RWT_pc.dat'))
            pc_sets = pcfile.pc_sets
            thicknesses, profiles = pc_sets[1]
            nprofiles = len(profiles)
            for n_p in range(nprofiles):
                p = profiles[n_p]
                cl = p[:,1]
                cl_max = cl+0.3*np.abs(cl)
                cl_min = cl-0.3*np.abs(cl)
                cl_new = cl_min+(cl_max-cl_min)*x[j]
                p[:,1] = cl_new
                profiles[n_p] = p
            pc_sets[1] = (thicknesses, profiles)
            pcfile.pc_sets = pc_sets
            pcName = "case_{:07d}_pc.dat".format(j)
            pcPath = os.path.join('data', pcName)
            pcfile.save(pcPath)
            htc.aero.pc_filename = r"./data/case_{:07d}_pc.dat".format(j)
            for k in range(nSeed):        
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()

    # def generate_cd(self, x, seed):
    #     n = self.n
    #     nSeed = self.nSeed
    #     htc = self.htc
    #     for j in range(n):
    #         pcfile = PCFile(r"data/DTU_10MW_RWT_pc.dat")
    #         pc_sets = pcfile.pc_sets
    #         thicknesses, profiles = pc_sets[1]
    #         nprofiles = len(profiles)
    #         for n_p in range(nprofiles):
    #             p = profiles[n_p]
    #             cd = p[:,2]
    #             cd_max = cd+0.2*np.abs(cd)
    #             cd_min = cd-0.2*np.abs(cd)
    #             cd_new = cd_min+(cd_max-cd_min)*x[j]
    #             p[:,2] = cd_new
    #             profiles[n_p] = p
    #         pc_sets[1] = (thicknesses, profiles)
    #         pcfile.pc_sets = pc_sets
    #         pcfile.save(r"./data/case_{:07d}_pc.dat".format(j))
    #         htc.aero.pc_filename = r"./data/case_{:07d}_pc.dat".format(j)
    #         for k in range(nSeed):        
    #             htc.wind.mann.create_turb_parameters[3] = seed[j,k]
    #             htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
    #             htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
    #             htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
    #             htc.set_name("case_%07d_%02d"%(j,k))
    #             htc.save()

    def generate_bladeIx(self, qms, seed):
        #%%  Strctural parameters
        x = qms.loc[:,"bladeIx"]
        n = self.n
        nSeed = self.nSeed
        htc = self.htc
        for j in range(n):
            stfile = StFile(r"data/DTU_10MW_RWT_Blade_st.dat")
            Ix = stfile.I_x()
            Ix_min = 0.5*Ix
            Ix_max = 1.5*Ix
            Ix_new = Ix_min+(Ix_max-Ix_min)*x[j]
            stfile.set_value(mset_nr=1, set_nr=1, I_x=Ix_new)
            stfile.save(r"./data/case_{:07d}_Blade_st.dat".format(j))
            htc.new_htc_structure.main_body__7.timoschenko_input.filename = r"./data/case_{:07d}_Blade_st.dat".format(j)
            for k in range(nSeed):        
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()

    def generate_towerIx(self, qms, seed):
        #%% Tower stiffness
        x = qms.loc[:,"towerIx"]
        n = self.n
        nSeed = self.nSeed
        htc = self.htc
        for j in range(n):
            stfile = StFile(r"data/DTU_10MW_RWT_Tower_st.dat")
            Ix = stfile.I_x()
            Ix_min = 0.5*Ix
            Ix_max = 1.5*Ix
            Ix_new = Ix_min+(Ix_max-Ix_min)*x[j]
            stfile.set_value(mset_nr=1, set_nr=1, I_x=Ix_new)
            stfile.set_value(mset_nr=1, set_nr=1, I_y=Ix_new)
            stfile.set_value(mset_nr=1, set_nr=1, I_p=2*Ix_new)
            stfile.save(r"./data/case_{:07d}_Tower_st.dat".format(j))
            htc.new_htc_structure.main_body__1.timoschenko_input.filename = r"./data/case_{:07d}_Tower_st.dat".format(j)
            for k in range(nSeed):        
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()

                        
    def remove_htc(self):
        n = self.n
        nSeed = self.nSeed
        for j in range(n):
            for k in range(nSeed):
                htcname = "case_%07d_%02d.htc"%(j,k)
                filename = os.path.join('htc', htcname)
                # filename = r"%s/htc/case_%07d_%02d.htc"%(folderName)
                os.remove(filename)

    def remove_pc(self):
        n = self.n
        for j in range(n):
            pcName = "case_{:07d}_pc.dat".format(j)
            pcPath = os.path.join('data', pcName)
            os.remove(pcPath)

    def remove_bladeSt(self):
        n = self.n
        for j in range(n):
            os.remove(r"./data/case_{:07d}_Blade_st.dat".format(j))

    def remove_towerSt(self):
        n = self.n
        for j in range(n):
            os.remove(r"./data/case_{:07d}_Tower_st.dat".format(j))

    def generate_mcs(self, X, seed):
        n = self.n
        nSeed = self.nSeed
        htc = self.htc
        # mean wind speed     
        u = 3+X.loc[:,"wsp"]*(27-3)
   
        for j in range(n):
            # wsp and ti
            htc.wind.wsp = u[j]
            sigma_min = np.max([0, 0.1*(u[j]-20)])
            sigma_max = 0.18*(6.8+0.75*u[j])
            sigma = sigma_min+(sigma_max-sigma_min)*X.loc[j,"ti"]
            ti = sigma/u[j]
            htc.wind.tint = ti
            htc.wind.wind_ramp_factor[2] = 7.8/u[j]
            turb_dx = u[j]*600/8192
            htc.wind.mann.box_dim_u[1] = turb_dx
            turb_dy = 178.3*1.15/32
            htc.wind.mann.box_dim_v[1] = turb_dy
            turb_dz = turb_dy
            htc.wind.mann.box_dim_w[1] = turb_dz
            # Cl
            pcfile = PCFile(os.path.join('data', 'DTU_10MW_RWT_pc.dat'))
            pc_sets = pcfile.pc_sets
            thicknesses, profiles = pc_sets[1]
            nprofiles = len(profiles)
            for n_p in range(nprofiles):
                p = profiles[n_p]
                cl = p[:,1]
                cl_max = cl+0.3*np.abs(cl)
                cl_min = cl-0.3*np.abs(cl)
                cl_new = cl_min+(cl_max-cl_min)*X.loc[j,"cl"]
                p[:,1] = cl_new
                profiles[n_p] = p
            pc_sets[1] = (thicknesses, profiles)
            pcfile.pc_sets = pc_sets
            pcName = "case_{:07d}_pc.dat".format(j)
            pcPath = os.path.join('data', pcName)
            pcfile.save(pcPath)
            htc.aero.pc_filename = r"./data/case_{:07d}_pc.dat".format(j)
            # Blade stiffness
            stfile = StFile(r"data/DTU_10MW_RWT_Blade_st.dat")
            Ix = stfile.I_x()
            Ix_min = 0.7*Ix
            Ix_max = 1.3*Ix
            Ix_new = Ix_min+(Ix_max-Ix_min)*X.loc[j,"bladeIx"]
            stfile.set_value(mset_nr=1, set_nr=1,I_x=Ix_new)
            stfile.save(r"./data/case_{:07d}_Blade_st.dat".format(j))
            htc.new_htc_structure.main_body__7.timoschenko_input.filename = r"./data/case_{:07d}_Blade_st.dat".format(j)
            #Tower stiffness
            stfile = StFile(r"data/DTU_10MW_RWT_Tower_st.dat")
            Ix = stfile.I_x()
            Ix_min = 0.7*Ix
            Ix_max = 1.3*Ix
            Ix_new = Ix_min+(Ix_max-Ix_min)*X.loc[j,"towerIx"]
            stfile.set_value(mset_nr=1, set_nr=1, I_x=Ix_new)
            stfile.set_value(mset_nr=1, set_nr=1, I_y=Ix_new)
            stfile.set_value(mset_nr=1, set_nr=1, I_p=2*Ix_new)
            stfile.save(r"./data/case_{:07d}_Tower_st.dat".format(j))
            htc.new_htc_structure.main_body__1.timoschenko_input.filename = r"./data/case_{:07d}_Tower_st.dat".format(j)

            for k in range(nSeed):
                htc.wind.mann.create_turb_parameters[3] = seed[j,k]
                htc.wind.mann.filename_u = r"./turb/case_{:07d}_{:02d}_turb_u.bin".format(j,k)
                htc.wind.mann.filename_v = r"./turb/case_{:07d}_{:02d}_turb_v.bin".format(j,k)
                htc.wind.mann.filename_w = r"./turb/case_{:07d}_{:02d}_turb_w.bin".format(j,k)
                htc.set_name("case_%07d_%02d"%(j,k))
                htc.save()

    
    def get_results(self, folderName, varName):
        n = self.n
        nSeed = self.nSeed
        bladeTowerDis_min = np.zeros((n,nSeed))
        Mx_blade_max = np.zeros((n,nSeed))
        My_blade_max = np.zeros((n,nSeed))
        Mres_blade_max = np.zeros((n,nSeed))
        Mx_tower_max = np.zeros((n,nSeed))
        My_tower_max = np.zeros((n,nSeed))
        Mres_tower_max = np.zeros((n,nSeed))
        for j in range(n):
            for k in range(nSeed):
                f = ReadHawc2(r"%s/%s/res/case_%07d_%02d"%(folderName, varName, j,k))
                Data= f()
                chInfo = f.ChInfo
                chName = chInfo[0]
                #unit = chInfo[1]
                #description = chInfo[2]
                df = pd.DataFrame(data=Data, columns=chName)
                # t = df.loc[:,'Time']
                bladeTowerDis = df.loc[:,'DLL inp   5:   1']
                # Flapwise moment blade root (Mx) -- out of plane moment
                Mx_blade = df.loc[:,'Mx coo: blade1']
                # Edgewise moment blade root (My) --- in plane moment
                My_blade = df.loc[:,'My coo: blade1']
                # Torque blade root (Mz) --- out of plane moment
                # resultant moment
                Mres_blade = np.sqrt(Mx_blade**2+My_blade**2)
                # Tower base foreaft moment (Mx)
                Mx_tower = df.loc[:,'Mx coo: tower']
                # Tower base side to side moment (My)
                My_tower = df.loc[:,'My coo: tower']
                # Tower base torque (Mz)
                # resultant moment
                Mres_tower = np.sqrt(Mx_tower**2+My_tower**2)
                bladeTowerDis_min[j,k] = np.min(bladeTowerDis)
                Mx_blade_max[j,k] = np.max(np.abs(Mx_blade)) 
                My_blade_max[j,k] = np.max(np.abs(My_blade)) 
                Mres_blade_max[j,k] = np.max(np.abs(Mres_blade)) 
                Mx_tower_max[j,k] = np.max(np.abs(Mx_tower)) 
                My_tower_max[j,k] = np.max(np.abs(My_tower)) 
                Mres_tower_max[j,k] = np.max(np.abs(Mres_tower)) 
                      
        bTD_median = np.median(bladeTowerDis_min, axis=1)
        Mx_blade_median = np.median(Mx_blade_max, axis=1)
        My_blade_median = np.median(My_blade_max, axis=1)
        Mres_blade_median = np.median(Mres_blade_max, axis=1)
        Mx_tower_median = np.median(Mx_tower_max, axis=1)
        My_tower_median = np.median(My_tower_max, axis=1)
        Mres_tower_median = np.median(Mres_tower_max, axis=1)
        # res dictionary
        res = {"bTD":bTD_median, "Mx_blade":Mx_blade_median, "My_blade":My_blade_median,
        "Mres_blade":Mres_blade_median, "Mx_tower":Mx_tower_median, "My_tower":My_tower_median,
        "Mres_tower":Mres_tower_median}
        return res

    def save_figues(self, varName, res):
        x, qms = self.read_qms('qms.h5', varName)
        # blade tower clearance
        plt.close('all')
        plt.figure(1, figsize=(8,5))
        plt.scatter(x, res["bTD"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Blade tower clearance (m)')
        plt.savefig(r'./Figures/{}_01.jpg'.format(varName))
        # Blade root flapwise moment Mx
        plt.figure(2, figsize=(8,5))
        plt.scatter(x, res["Mx_blade"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Blade root flapwise moment Mx (KN-m)')
        plt.savefig(r'./Figures/{}_02.jpg'.format(varName))
        # Blade root edgewise moment My
        plt.figure(3, figsize=(8,5))
        plt.scatter(x, res["My_blade"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Blade root edgewise moment My (KN-m)')
        plt.savefig(r'./Figures/{}_03.jpg'.format(varName))
        # Blade root resultant moment Mres
        plt.figure(4, figsize=(8,5))
        plt.scatter(x, res["Mres_blade"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Blade root resultant moment Mres (KN-m)')
        plt.savefig(r'./Figures/{}_04.jpg'.format(varName))
        # Tower base foreaft moment Mx
        plt.figure(5, figsize=(8,5))
        plt.scatter(x, res["Mx_tower"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Tower base foreaft moment Mx (KN-m)')
        plt.savefig(r'./Figures/{}_05.jpg'.format(varName))
        # Tower base side-to-side moment My
        plt.figure(6, figsize=(8,5))
        plt.scatter(x, res["My_tower"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Tower base side to side moment My (KN-m)')
        plt.savefig(r'./Figures/{}_06.jpg'.format(varName))
        # Tower base resultant moment Mres
        plt.figure(7, figsize=(8,5))
        plt.scatter(x, res["Mres_tower"])
        plt.xlabel('{}'.format(varName))
        plt.ylabel('Tower base resultant moment Mres (KN-m)')
        plt.savefig(r'./Figures/{}_07.jpg'.format(varName))


        
  
if __name__ == "__main__":
    print("okay")