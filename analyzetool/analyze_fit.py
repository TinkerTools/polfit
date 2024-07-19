import numpy as np
import os
import sys
import math, copy
from time import gmtime, strftime
import pandas as pd
import filecmp
from analyzetool import auxfitting, prmedit, process
import analyzetool.gas as gasAnalyze
import analyzetool.liquid as liqAnalyze
from analyzetool.process import save_pickle
from analyzetool.process import load_pickle
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec


datadir = "/home/roseane/HIPPO/small_molecules/org_molecules/reference-data"

prmdir = f"{datadir}/prmfiles"

dbase_info = f"{datadir}/database-info"
molinfo = load_pickle(f"{dbase_info}/molinfo_dict.pickle")
database_full = load_pickle(f"{dbase_info}/full_database.pickle")
desres_dimer_info = load_pickle(f"{dbase_info}/desres_dimer_info.pickle")
molIDs = load_pickle(f"{dbase_info}/molIDs.pickle")
watercid = 962

ref_data = datadir
smallmoldir = "~/HIPPO/small_molecules"

refpath = f"{smallmoldir}/org_molecules/fitting-2"
testfit = f"{refpath}/test-fit"
dbases = ["HB375x10","HB300SPXx10","R739x5","SH250x10"]
class FITANALYSIS(object):
    def __init__(self,n,fitpath=refpath,testfit=testfit,use_dict=True,use_fitprm=False):
        self.path = f"{testfit}/{n}"
        self.fitpath = f"{fitpath}/{n}"
        self.n = n
        self.cid = database_full[n][-1]
        self.liqres = []
        molfit = auxfitting.Auxfit(testfit,n)

        molfit.datadir = ref_data
        molfit.smallmoldir = smallmoldir
        molfit.prepare_directories()
        molfit.process_prm()

        prmfit = f"{self.fitpath}/fit_results/{self.n}-latest.prm"

        if use_dict:
            preserve_terms = ['opbend','strbnd','torsion','bndcflux','angcflux']
            files = next(os.walk(self.fitpath))[2]
            dictfn = np.array([f for f in files if 'newprms' in f])
            modtim = [os.path.getmtime(f"{self.fitpath}/{f}") for f in dictfn]
            modtim = np.array(modtim)
            inds = np.argsort(modtim)

            if len(dictfn) > 0:
                ndict = f"{self.fitpath}/{dictfn[inds[-1]]}"

                newdict = load_pickle(ndict)
                print(f"Loading previous prmdict: {dictfn[inds[-1]]}\n")

                for term in preserve_terms:
                    newdict[term] = copy.deepcopy(molfit.prmdict[term])

                molfit.prmdict = copy.deepcopy(newdict)

        if use_fitprm and os.path.isfile(prmfit):
            molfit.process_prm(prmfit)
        
        molfit.build_prm_list()
        molfit.make_key()
        molfit.initpotrms = molfit.get_potfit()

        molfit.liquid_fitproperties()

        if os.path.isfile(f"{ref_data}/qm-calc/{n}/sapt-res-water+mol.npy"):
            molfit.prepare_opt_sapt_dimers()
        if os.path.isdir(f"{ref_data}/qm-calc/{n}/sapt_dimers"):
            molfit.prepare_sapt_dimers()
        if os.path.isdir(f"{ref_data}/qm-calc/{n}/clusters"):
            molfit.prepare_cluster()
        if os.path.isdir(f"{ref_data}/qm-calc/{n}/ccsdt_dimers"):
            molfit.prepare_ccsdt_dimers()

        self.resdir = f"{testfit}/results/{n}"
        if not os.path.isdir(self.resdir):
            os.makedirs(self.resdir)

        self.molfit = molfit

    def run_analysis(self):
        self.rms = self.molfit.get_potfit()
        self.err_pol = self.molfit.get_polarize()
        self.molpol = self.molfit.molpol
        # if self.molfit.do_dimers:
        #     print("OPT SAPT DIMERS")
        #     self.opt_dimer, self.err_opt_dimer = self.molfit.compute_water_dimers()
        if self.molfit.do_ccsdt_dimers:
            print("CCSD(T) DIMERS")
            ccsdt_dimer1, self.err_ccsdt_dimer = self.molfit.compute_ccsdt_dimer()
            self.ccsdt_dimer = {}
            for i,nm in enumerate(self.molfit.ccsdt_dimers):
                if len(self.molfit.ccsdt_dimers) == 1:
                    self.ccsdt_dimer[nm] = ccsdt_dimer1
                else:
                    self.ccsdt_dimer[nm] = ccsdt_dimer1[i]
        if self.molfit.do_sapt_dimers:
            print("SAPT DIMERS")
            sapt_dimer1, self.err_sapt_dimer = self.molfit.compute_sapt_dimers()
            
            self.sapt_dimer = {}
            for i,nm in enumerate(self.molfit.sapt_dimers):
                if len(self.molfit.sapt_dimers) == 1:
                    self.sapt_dimer[nm] = sapt_dimer1
                else:
                    self.sapt_dimer[nm] = sapt_dimer1[i]
        if self.molfit.do_clusters:
            print("CLUSTER")
            self.clusters, self.err_cluster = self.molfit.compute_cluster()

        self.liquid = 0

    def reload_data(self,use_dict=True,use_fitprm=False,prmfile=""):
        if use_dict:
            # PRESERVE SOME PARAMETERS FROM ORIGINAL PRMFILE4
            preserve_terms = ['opbend','strbnd','torsion','bndcflux','angcflux']
            files = next(os.walk(self.fitpath))[2]
            dictfn = np.array([f for f in files if 'newprms' in f])
            modtim = [os.path.getmtime(f"{self.fitpath}/{f}") for f in dictfn]
            modtim = np.array(modtim)
            inds = np.argsort(modtim)

            if len(dictfn) > 0:
                ndict = f"{self.fitpath}/{dictfn[inds[-1]]}"

                newdict = load_pickle(ndict)
                print(f"Loading previous prmdict: {dictfn[inds[-1]]}\n")

                for term in preserve_terms:
                    newdict[term] = copy.deepcopy(self.molfit.prmdict[term])

                self.molfit.prmdict = copy.deepcopy(newdict)

        if use_fitprm and os.path.isfile(prmfit):
            prmfit = f"{self.fitpath}/fit_results/{self.n}-latest.prm"
            self.molfit.process_prm(prmfit)
        if len(prmfile) > 0:
            if os.path.isfile(prmfile):
                self.molfit.process_prm(prmfile)

    def run_liquid_analysis(self,liqpath='ref_liquid',gaslog='gas.log',eq=1000,simd=""):
        n = self.n
        info = molinfo[n]

        if len(simd) == 0:
            simd = f"{self.fitpath}/{liqpath}"
        
        if os.path.isdir(simd):
            t = info[0]
            ref1 = info[1]
            ref2 = info[2]
            ref3 = info[3]
            ref4 = info[4]
            ref5 = info[5]


            molpol = self.molfit.molpol.mean()
            try:
                liquid = liqAnalyze.Liquid(simd,'liquid.xyz',self.molfit.Natoms,t,eq,
                                                        logfile='liquid.log',analyzelog=None,
                                                        sysinfo=None,gaslog=gaslog,molpol=molpol)
            except:
                return

            if os.path.isfile(f"{simd}/analysis.log"):
                liquid.get_dielectric(f"{simd}/analysis.log",molpol)
            
            try:
                d = liquid.avgRho*1000
                hv = 4.184*liquid.HV
                eps = liquid.dielectric
                kappa = liquid.kappa/100
                alphaT = (1e3)*liquid.alpha
                rlen = liquid.Rhos.shape[0]
            except:
                return
                
            derr = 100*(d-ref1)/ref1
            if ref2 != -1:
                hve = 100*(hv-ref2)/ref2
            else:
                hve = 0

            if ref3 != -1:
                epse = 100*(eps-ref3)/ref3
            else:
                epse = 0

            if ref4 != -1:
                kte = 100*(kappa-ref4)/ref4
            else:
                kte = 0

            if ref5 != -1:
                ape = 100*(alphaT-ref5)/ref5
            else:
                ape = 0
                
            self.liqref = [ref1,ref2,ref3,ref4,ref5]
            self.liqerr = [derr,hve,epse,kte,ape]
            self.liqres = [d,hv,eps,kappa,alphaT]
            self.rlen = rlen
            self.liquid = liquid

    def make_fitsummary(self,writeout=True,printout=False,dumpres=True,run_analysis=False,reload_data=False,use_dict=True,use_fitprm=False,prmfile="",
                        liqpath='ref_liquid',gaslog='gas.log',eq=1000):
        n = self.n
        if reload_data:
            self.reload_data(use_dict,use_fitprm,prmfile)
        if run_analysis:
            self.run_analysis()

        rms = self.rms
        refpol = self.molfit.refmolpol
        molpol = self.molfit.molpol
        diff = np.zeros(3)
        for k in range(3):
            if np.sign(refpol[k]) != np.sign(molpol[k]):
                diff[k] += np.abs(refpol[k]) - molpol[k]
            else:
                diff[k] += refpol[k] - molpol[k]

        diff = np.abs(diff)
        mdiff = np.abs(np.abs(refpol.mean())-np.abs(molpol.mean()))

        textres = f"#{n:<d} {database_full[n][0].capitalize():<s} {database_full[n][1]}  CID: {database_full[n][-1]}\n\n"
        textres += f"ref molpol {refpol[0]:7.2f} {refpol[1]:7.2f} {refpol[2]:7.2f}, avg {refpol.mean():7.2f} \n"
        textres += f"    molpol {molpol[0]:7.2f} {molpol[1]:7.2f} {molpol[2]:7.2f}, avg {molpol.mean():7.2f} \n"
        textres += f"rms molpol {diff[0]:7.2f} {diff[1]:7.2f} {diff[2]:7.2f}, avg {mdiff:7.2f}\n"
        textres += f"\nMonomer potential fitting RMS: {rms:<7.2f}\n\n"

        tres1 = ""
        if self.molfit.do_ccsdt_dimers:
            tres1 = "##Dimer results - Fitting to QM datasets##\n\n"
            for i,nm in enumerate(self.molfit.ccsdt_dimers):
                ref = self.molfit.ccsdt_dimers_ref[nm]
                res = self.ccsdt_dimer[nm]

                natm = self.molfit.Natoms
                if 'water' in nm:
                    cm,cm1,cm2 = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,3])
                else:
                    cm,cm1,cm2 = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,natm])
                dist = np.linalg.norm(cm2-cm1,axis=1)
                dist = dist[self.molfit.ccsdt_dimers_indx[nm]]   

                indx = np.argsort(dist)
                dist = dist[indx]
                res = res[indx]
                ref = ref[indx]

            #     print(nm)
                tres1 += f"{nm}, energy values in kcal/mol\n"

                headers = ["CM-CM (A)","Reference","HIPPO res","Abs diff"]
                if 'DES' not in nm:
                    tres1 += f'\n    {headers[0]:10s} {headers[1]:10s} {headers[2]:10s} {headers[3]:10s}\n'
                    for kk,val in enumerate(res):
            #             print(f"{dist[kk]:10.3f} {ref[kk]:10.3f} {val:10.3f} {val-ref[kk]:10.4f}")
                        tres1 += f"{dist[kk]:10.3f} {ref[kk]:10.3f} {val:10.3f} {val-ref[kk]:10.4f}\n"
                err = np.abs(res-ref)
                nerr = err[err > 1]

                valerr = self.err_ccsdt_dimer[i]

                headers = ["   MAE","Std error","max error","#points","#count[err > 1]"]
                tres1 += f'\n    {headers[0]:10s} {headers[1]:10s} {headers[2]:10s} {headers[3]:8s} {headers[4]:9s}\n'
            #     print("Averages")
                tres1 += f"    {err.mean():7.3f} {err.std():10.3f} {err.max():10.4f} {err.shape[0]:8d} {nerr.shape[0]:10d}\n\n"

        tres2 = ""
        if n in molinfo.keys():
            info = molinfo[n]
            tres2 = f"Liquid {database_full[n][0].capitalize():<s} @ {info[0]:<6.2f} K\n"
            # print(f"#{n:<d} {database_full[n][0].capitalize():<s} @ {info[1]:<6.2f} K")
            line0 =  f"  {'Density':8s} {'Ref-Dens':8s} {'%err':>5s} {'HV':>6s} {'Ref-HV':>7s} {'%err':>5s} " 
            line0 += f"{'Dielec':>7s} {'Ref-D':>6s} {'%err':>5s} {'kappa':>6s} {'Ref-k':>6s} {'%err':>5s} "
            line0 += f"{'alphaT':6s} {'Ref-aT':>6s} {'%err':>6s} {'#nFrm':>7s}\n"
            tres2 += line0
            if len(self.liqres) == 0:
                self.run_liquid_analysis(liqpath,gaslog,eq)
            if len(self.liqres) != 0:
                [ref1,ref2,ref3,ref4,ref5] = self.liqref 
                [derr,hve,epse,kte,ape] = self.liqerr 
                [d,hv,eps,kappa,alphaT] = self.liqres
                rlen = self.rlen
                
                line =  f"{d:9.2f} {ref1:8.2f} {derr:5.1f} {hv:7.2f} {ref2:7.2f} {hve:5.1f} " 
                line += f"{eps:7.2f} {ref3:6.2f} {epse:5.1f} {kappa:6.2f} {ref4:5.2f} {kte:6.1f} "
                line += f"{alphaT:6.2f} {ref5:5.2f} {ape:6.1f} {rlen:>7d}\n"

                tres2 += line

        ## Cluster
        tres3 = ""
        comps = [" ","Elec","Exch","Indc","Disp","Total"]
        if self.molfit.do_clusters:
            tres3 = "\n Results for cluster structures, energies in kcal/mol\n\n"
            ll = [f"{hh:8s}" for hh in comps]
            tres3 += " "+ "".join(ll)+'\n'
            for i,nm in enumerate(self.molfit.cluster_ref.keys()):
                ref = self.molfit.cluster_ref[nm]
                res = self.clusters[i]
                line = [f"{ref[i]:8.2f}" for i in range(len(ref))]
                line = f"{nm:6s}"+"".join(line)+'\n'
                line2 = [f"{res[i]:8.2f}" for i in range(len(res))]
                line += f"{'':6s}"+"".join(line2)+'\n\n'
                tres3 += line

        if writeout:
            fname = f"{self.resdir}/fitting_summary.txt"
            with open(fname,'w') as thefile:
                thefile.write(textres+tres1+tres2+tres3)
        if printout:
            print(textres+tres1+tres2+tres3)
        if dumpres:
            savedir = f"{self.path}/dump-results"
            if not os.path.isdir(savedir):
                os.mkdir(savedir)

            origprm = f"{datadir}/prmfiles/{n}.prm"
            prm1 = prmedit.process_prm(origprm)
            mfacts = prm1['multipole_factors']
            prmedit.write_prm(self.molfit.prmdict,f"{savedir}/{n}.prm",mfacts)

            num = 1
            files = next(os.walk(savedir))[1]
            files = [a for a in files if 'res' in a]
                        
            nums = [0]
            comp = False
            for a in files:
                nk1 = int(a.split('-')[-1])
                nums.append(nk1)
                nprmfn = f"{savedir}/{a}/{n}.prm"
                if os.path.isfile(nprmfn):
                    comp = filecmp.cmp(f"{savedir}/{n}.prm",nprmfn)
                    if comp:
                        fnm = f"res-{nums[-1]}"
                        break
            
            if not comp:
                nums = sorted(nums)
                fnm = f"res-{nums[-1]+1}"

            if not os.path.isdir(f"{savedir}/{fnm}"):
                os.mkdir(f"{savedir}/{fnm}")

            os.system(f"mv {savedir}/{n}.prm {savedir}/{fnm}/")
            
            nliq,fnliq = process.next_folder_number(f"{savedir}/{fnm}",'liqdata') 
            if self.liquid:
                save_pickle(self.liquid,f"{fnliq}")
            os.system(f"cp {self.resdir}/* {savedir}/{fnm}/")

            save_pickle(self.molfit,f"{savedir}/{fnm}/results.pickle")
            return f"{savedir}/{fnm}"

    def make_all_plots(self,save=True,show=False):
        self.make_ccsdt_plots(save,show)
        # if not self.molfit.do_sapt_dimers:
        self.make_ccsdt_plots_desres(save,show,todos=False)
        self.make_sapt_plots(save,show)
        self.make_sapt_plot_desres(save,show)
        # self.make_opt_sapt_plots(save,show)
        self.benzene_parallel_disp(save,show)
        return
    
    def make_ccsdt_plots(self,save=True,show=False):
        if not self.molfit.do_ccsdt_dimers:
            return
        marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o')
        marker_style2 = dict(color='black', linestyle='--',linewidth=2,marker='o')
        marker_style3 = dict(color='red', marker='o')

        nplots = 0

        plotnm = []
        results = []
        nall = len(self.molfit.ccsdt_dimers)
        ncols = []
        for i,nm in enumerate(self.molfit.ccsdt_dimers):
            if 'DES' not in nm:
                nplots += 1
                plotnm.append(nm)
                db = nm.split('_')[0]
                if db in dbases:
                    nconf = int(db.split('x')[-1])
                    res1 = self.ccsdt_dimer[nm]
                    ccol = int(len(res1)/nconf)
                    if ccol == 2:
                        ncols.append(4)
                    elif ccol == 3:
                        ncols.append(6)
                    else:
                        ncols.append(2)
        
        if len(plotnm) == 0:
            return

        ncol = np.max(ncols)
        
        sizes = [8,16,24,28,32,40]
        nn = nplots % 2
        n1 = int(np.ceil(nplots/2))

        if n1 > 0:
            f, ax = plt.subplots(n1,2,figsize=(24, sizes[n1-1]))
        else:
            return

        if n1 == 1:
            ax = [ax]

        for kk in range(n1):
            for ii in range(2):
                try:
                    nm = plotnm[kk*2+ii]
                except:
                    break
                
                sapt_ref1 = self.molfit.ccsdt_dimers_ref[nm]
                res1 = self.ccsdt_dimer[nm]


                natm = self.molfit.Natoms
                if 'water' in nm:
                    cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,3])
                else:
                    cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,natm])
                dist1 = np.linalg.norm(cm2-cm1,axis=1)
                dist1 = dist1[self.molfit.ccsdt_dimers_indx[nm]]   

                indx = np.argsort(dist1)
                dist = dist1[indx]
                res = res1[indx]
                sapt_ref = sapt_ref1[indx]

                db = nm.split('_')[0]

                if "water" in nm:
                    title = f"Water + #{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db}"
                else:
                    title = f"#{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db}"

                doplot = True
                nsp = int(ncol/2)
                xlabel = r"CM-CM distance (${\AA}$)"
                if db in dbases:
                    nconf = int(db.split('x')[-1])
                    ncf = int(len(sapt_ref)/nconf)

                    if ncf > 1:
                        min1 = np.min(sapt_ref1)
                        min2 = np.min(res1)
                        emin = np.min([min1,min2])

                        max1 = np.max(sapt_ref1)
                        max2 = np.max(res1)
                        emax = np.max([max1,max2])

                        res = res1[:nconf]
                        sapt_ref = sapt_ref1[:nconf]
                        dist = dist1[:nconf]
                        ax = plt.subplot2grid((n1, ncol), (kk, ii*nsp), colspan=1)
                        ax.scatter(dist,sapt_ref,**marker_style1)
                        ax.plot(dist,sapt_ref,label="CCSD(T) Total",**marker_style1)
                        ax.scatter(dist,res,**marker_style2)
                        ax.plot(dist,res,label="HIPPO Total",**marker_style2)
                        # xlabel = "# Points (not CM-CM distance)"
                        doplot = False

                        # axs = [plt.subplot2grid((n1, ncol), (kk, ii*nsp+pi), colspan=1,sharey=ax) for pi in range(1,ncf)]
                        xx = ax.get_xticks()
                        yy = ax.get_yticks()

                        if yy[0] > emin:
                            yy[0] = emin-0.2
                        if yy[-1] < emax:
                            yy[-1] = emax+0.2

                        # Set major ticks for y axis
                        nticks = 5
                        if emax - emin < 15:
                            nticks = 2
                        if emax - emin < 8:
                            nticks = 1
                        if emax - emin < 4:
                            nticks = 0.5
                        if emax - emin < 2:
                            nticks = 0.25
                            yy[-1] = yy[0]+2
                        major_yticks = np.arange(yy[0], yy[-1], nticks)

                        # Specify tick label size
                        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
                        # Suppress minor tick labels
                        ax.set_yticks(major_yticks)
                        yy = ax.get_yticks()
                        if ncf >= 2:
                            axs = plt.subplot2grid((n1, ncol), (kk, ii*nsp+1), colspan=1)
                            dist = dist1[nconf:2*nconf]
                            res = res1[nconf:2*nconf]
                            sapt_ref = sapt_ref1[nconf:2*nconf]

                            axs.scatter(dist,sapt_ref,**marker_style1)
                            axs.plot(dist,sapt_ref,label="CCSD(T) Total",**marker_style1)
                            axs.scatter(dist,res,**marker_style2)
                            axs.plot(dist,res,label="HIPPO Total",**marker_style2)

                            plt.subplots_adjust(wspace=0.0,hspace=0.15)
                            y2 = yy.astype(int)
                            # axs.axes.get_yaxis().set_ticks(yy)
                            axs.set_yticks(major_yticks)
                            # ax.axes.get_yaxis().set_ticks(yy)
                            # ax.axes.get_yaxis().set_ticks(y2)
                            # axs.axes.get_yaxis().set_ticks(y2)

                            xx = axs.get_xticks()
                            axs.axes.get_xaxis().set_ticks(xx[1::2])
                            
                            axs.yaxis.set_minor_locator(AutoMinorLocator(n=2))
                            # axs.minorticks_off()
                            axs.grid('x',which='major')
                            axs.grid('y',which='both')
                            # axs.axes.get_yaxis().set_ticks(yy[-1:])
                            
                            [t.set_color('white') for t in axs.yaxis.get_ticklabels()]

                            if ncf == 2:
                                axs.set_title(title,fontsize=20,loc='right')
                                axs.set_xlabel(xlabel,fontsize=22,loc='left')
                            else:
                                axs.set_xlabel(xlabel,fontsize=22)
                        if ncf == 3:
                            ax2 = plt.subplot2grid((n1, ncol), (kk, ii*nsp+2), colspan=1)
                            dist = dist1[2*nconf:]
                            res = res1[2*nconf:]
                            sapt_ref = sapt_ref1[2*nconf:]

                            ax2.scatter(dist,sapt_ref,**marker_style1)
                            ax2.plot(dist,sapt_ref,label="CCSD(T) Total",**marker_style1)
                            ax2.scatter(dist,res,**marker_style2)
                            ax2.plot(dist,res,label="HIPPO Total",**marker_style2)

                            plt.subplots_adjust(wspace=0.0)
                            # ax2.axes.get_yaxis().set_ticks(yy)
                            # ax2.axes.get_yaxis().set_ticks(y2)

                            xx = ax2.get_xticks()
                            ax2.axes.get_xaxis().set_ticks(xx[1::2])
                            ax2.set_yticks(major_yticks)

                            ax2.yaxis.set_minor_locator(AutoMinorLocator(n=2))
                            # ax2.minorticks_off()
                            ax2.grid('x',which='major')
                            ax2.grid('y',which='both')
                            [t.set_color('white') for t in ax2.yaxis.get_ticklabels()]

                            ax2.set_title(title,fontsize=20,loc='right')
                            
                        # for ff in range(nconf,len(sapt_ref),nconf):
                        #     # ax1 = plt.subplot2grid((n1, ncol), (kk, ii*nsp+pi), colspan=1, sharey=ax)
                        #     # dist = np.arange(nconf,dtype=int)+1+ff
                        #     dist = dist1[ff:ff+nconf]
                        #     res = res1[ff:ff+nconf]
                        #     sapt_ref = sapt_ref1[ff:ff+nconf]

                        #     axs[pi].scatter(dist,sapt_ref,**marker_style1)
                        #     axs[pi].plot(dist,sapt_ref,label="CCSD(T) Total",**marker_style1)
                        #     axs[pi].scatter(dist,res,**marker_style2)
                        #     axs[pi].plot(dist,res,label="HIPPO Total",**marker_style2)
                        #     xlabel = "# Points (not CM-CM distance)"
                        #     pi += 1
                            
                    

                if doplot:
                    
                    ax = plt.subplot2grid((n1, ncol), (kk, nsp*ii), colspan=nsp)
                    ax.scatter(dist,sapt_ref,**marker_style1)
                    ax.plot(dist,sapt_ref,label="CCSD(T) Total",**marker_style1)
                    ax.scatter(dist,res,**marker_style2)
                    ax.plot(dist,res,label="HIPPO Total",**marker_style2)

                    ax.set_title(title,fontsize=20,loc='right')
                    ax.set_xlabel(xlabel,fontsize=25)

                    xx = ax.get_xticks()
                    yy = ax.get_yticks()
                    # Set major ticks for y axis
                    nticks = 5
                    if np.abs(yy[-1]-yy[0]) < 15:
                        nticks = 2
                    if np.abs(yy[-1]-yy[0]) < 8:
                        nticks = 1
                    if np.abs(yy[-1]-yy[0]) < 4:
                        nticks = 0.5
                    if np.abs(yy[-1]-yy[0]) < 2.5:
                        nticks = 0.5
                        yy[-1] = yy[0]+2.5

                    major_yticks = np.arange(yy[0], yy[-1], nticks)

                    # Specify tick label size
                    ax.tick_params(axis = 'both', which = 'major', labelsize = 22)
                    # Suppress minor tick labels
                    # ax.set_yticks(major_yticks)
                    ax.axes.get_yaxis().set_ticks(major_yticks)

                if ii == 0:
                    ax.set_ylabel('Energy (kcal/mol)',fontsize=25)

                # axs.axes.get_yaxis().set_ticks([])
        #         ax.set_xticks()
                
                ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
                # ax.minorticks_off()
                ax.grid('x',which='major')
                ax.grid('y',which='both')
        #       ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))

                if (kk*2+ii) == len(plotnm)-1:
                    break


        
        # if kk == n1-1 or n1 == 1:
        # if n1 == 1:
        #     ax[kk][0].legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(1.45, -.20),ncol=2, fancybox=True, shadow=True)
        # else:
        if ii == 0:
            ax.legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(1.45, -.20),ncol=2, fancybox=True, shadow=True)
        else:
            ax.legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(0.3, -.25),ncol=2, fancybox=True, shadow=True)

        # plt.grid(True)
        plt.subplots_adjust(wspace=0.20,hspace=0.3)

        if save:
            plt.savefig(f'{self.resdir}/ccsdt_dimers.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')
        if show:
            plt.show()

        plt.close()
        return
    
    def make_ccsdt_plots_desres(self,save=True,show=False,todos=False,maxp=7):
        if not self.molfit.do_ccsdt_dimers:
            return

        marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o')
        marker_style2 = dict(color='black', linestyle='--',linewidth=2,marker='o')

        sizes = [8,16,32,36,38,40]
        for r in range(6,35):
            val = sizes[r-1] + 6
            sizes += [val]
        for i,nm in enumerate(self.molfit.ccsdt_dimers):
            sapt_ref = self.molfit.ccsdt_dimers_ref[nm]

            if nm in self.molfit.sapt_dimers:
                continue

            if 'DES' not in nm:
                continue
            if 'water' in nm:
                cid2 = watercid
            else:
                cid2 = self.cid

            res = self.ccsdt_dimer[nm]

            natm = process.count_atoms(f"{self.path}/liquid/gas.xyz")
            if 'water' in nm:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,3])
            else:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,natm])
            dist = np.linalg.norm(cm2-cm1,axis=1)
            dist = dist[self.molfit.ccsdt_dimers_indx[nm]]

            ranges = []
            nstart = 0
            for gid,dt in desres_dimer_info[self.cid][cid2].items():
                if dt[0] == 'md_dimer' or dt[0] == 'qm_opt_dimer':
                    ranges.append([nstart,nstart+dt[1]])
                nstart+=dt[1]

            nplots = len(ranges)

            if nplots == 0:
                continue

            nn = nplots % 3
            n1 = int(np.ceil(nplots/3))
            
            sizes = [8,16,32,36,38,40]
            if n1-1 > len(sizes):
                df = sizes[-1]
                for j in range(n1-len(sizes)):
                    sizes += [df+8*(j+1)]  
            
            if todos:
                maxp = n1

            nn1 = np.arange(n1)
            if n1 > maxp:
                nn2 = maxp
            else:
                nn2 = n1
            
            if n1 > 0:
                f, ax = plt.subplots(nn2,3,figsize=(36, sizes[nn2-1]))
            else:
                return

            if n1 == 1:
                ax = [ax]
            
            for kk,pp in enumerate(nn1[:maxp]):
                for ii in range(3):
                    try:
                        n0,nf = ranges[kk*2+ii]
                    except:
                        continue
                    m1 = self.molfit.ccsdt_dimers_indx[nm] > n0-1
                    m2 = self.molfit.ccsdt_dimers_indx[nm] < nf
                    m4 = m1&m2
                    m5 = dist < 7.2
                    m3 = m4&m5
                    indx = self.molfit.ccsdt_dimers_indx[nm][m3]

                    ax[kk][ii].plot(dist[m3],sapt_ref[m3],**marker_style1,label=f"CCSD(T)/CBS Total")
                    ax[kk][ii].plot(dist[m3],res[m3],label="HIPPO Total",**marker_style2)

                    db = nm.split('_')[0]
                    if "water" in nm:
                        title = f"Water + #{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"
                    else:
                        title = f"#{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"

                    if ii == 0:
                        ax[kk][ii].set_ylabel('Energy (kcal/mol)',fontsize=25)                           

                    if kk == 0:
                        yy = ax[kk][ii].get_yticks()
                        if nplots == 1 and ii == 0:
                            ax[kk][ii].set_yticks(yy[:-1])
                            ax[kk][ii].set_title(title,fontsize=22)
                        elif nplots == 2 and ii == 1:
                            ax[kk][ii].set_yticks(yy[:-1])
                            ax[kk][ii].set_title(title,fontsize=22)
                        elif nplots > 2 and ii == 2:
                            ax[kk][ii].set_yticks(yy[:-1])
                            ax[kk][ii].set_title(title,fontsize=22)

            #         ax.set_xticks()
                    ax[kk][ii].minorticks_off()
                    ax[kk][ii].yaxis.set_minor_locator(AutoMinorLocator(n=2))
            #         ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
                    ax[kk][ii].grid(which='both')

            if kk == nn2-1:
                ax[kk][0].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=22)
                ax[kk][1].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=22)
                ax[kk][2].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=22)
                ax[kk][0].legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(2.0, -.25),ncol=2, fancybox=True, shadow=True)
            
            plt.subplots_adjust(wspace=0.15,hspace=0.15)
            plt.grid(True)

            if save:
                plt.savefig(f'{self.resdir}/{nm}.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')
            
            if show:
                plt.show()

            plt.close()

     
    def make_sapt_plots(self,save=True,show=False):
        if not self.molfit.do_sapt_dimers:
            return
        plt.rcParams.update({'font.size': 22})

        marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o')
        marker_style2 = dict(color='black', linestyle='--',linewidth=2,marker='o')

        marker_style3 = dict(color='red', linestyle='-',linewidth=2,marker='v')
        marker_style4 = dict(color='red', linestyle='--',linewidth=2,marker='v')

        marker_style5 = dict(color='blue', linestyle='-',linewidth=2,marker='D')
        marker_style6 = dict(color='blue', linestyle='--',linewidth=2,marker='D')

        marker_style7 = dict(color='green', linestyle='-',linewidth=2,marker='x')
        marker_style8 = dict(color='green', linestyle='--',linewidth=2,marker='x')

        marker_style9 = dict(color='magenta', linestyle='-',linewidth=2,marker='s')
        marker_style10 = dict(color='magenta', linestyle='--',linewidth=2,marker='s')

        compnts = ['electrostatics','exchange','induction','dispersion','total']
        xlabel = r"CM-CM distance (${\AA}$)"

        plotnm = []
        nall = len(self.molfit.sapt_dimers)
        nwater = 0
        for nm in self.molfit.sapt_dimers:
            if 'water' in nm:
                nwater += 1
        nsort = np.argsort(self.molfit.sapt_dimers)    
        for i,kk in enumerate(nsort):
            nm = self.molfit.sapt_dimers[kk]
            if 'DES' not in nm:
                # if nwater > 1 and 'water+mol' in nm:
                #     continue
                plotnm.append(nm)
            if 'DES' in nm and 'sapt' in nm:
                plotnm.append(nm)
        n1 = len(plotnm)
        sizes = [8,16,32,36,38,40]
        if n1 > len(sizes):
            df = sizes[-1]
            for j in range(n1-len(sizes)+2):
                sizes += [df+8*(j+1)]         

        if n1 > 7:
            nn1 = np.random.choice(np.arange(n1),7,False)
            nn2 = 7
        else:
            nn1 = np.arange(n1)
            nn2 = n1
        
        if n1 > 0:
            f, ax = plt.subplots(nn2,3,figsize=(36, sizes[nn2-1]))
        else:
            return

        if nn2 == 1:
            ax = [ax]
        
        for kk,pp in enumerate(nn1):
            try:
                nm = plotnm[pp]
            except:
                break

            sapt_ref1 = self.molfit.sapt_dimers_ref[nm]
            res1 = self.sapt_dimer[nm]

            natm = self.molfit.Natoms
            if 'water' in nm:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,3])
            else:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,natm])
            
            dist1 = np.linalg.norm(cm2-cm1,axis=1)
            dist1 = dist1[self.molfit.sapt_dimers_indx[nm]]

            indx = np.argsort(dist1)
            dist = dist1[indx]
            res = res1[indx]
            sapt_ref = sapt_ref1[indx]

            xlabel = r"CM-CM distance (${\AA}$)"
            title = nm
            db = nm.split('_')[0]
            if 'water+mol' in nm:
                title = f"Water + #{self.n} {database_full[self.n][0].capitalize():<s} Dimer - randomly generated conformations"               
            elif "water" in nm:
                    title = f"Water + #{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"
            else:
                title = f"#{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"
            
            if 'water+mol' in nm and nn2 == 1:
                dist = np.arange(dist1.shape[0],dtype=int)+1

            ax[kk][0].plot(dist,res[:,-1],**marker_style2)    
            ax[kk][0].plot(dist,sapt_ref[:,-1],**marker_style1)
            ax[kk][1].plot(dist,res[:,0],**marker_style4)
            ax[kk][1].plot(dist,sapt_ref[:,0],**marker_style3)
            ax[kk][1].plot(dist,res[:,3],**marker_style8)
            ax[kk][1].plot(dist,sapt_ref[:,3],**marker_style7)
            ax[kk][2].plot(dist,res[:,2],**marker_style10)
            ax[kk][2].plot(dist,sapt_ref[:,2],**marker_style9)

            ax[kk][0].set_ylabel('Energy (kcal/mol)',fontsize=25)

            ax[kk][2].set_title(title,fontsize=22,loc='right')            
            
            xlim = ax[kk][0].get_xlim()
            xx = ax[kk][0].get_xticks()
            xmax = 7
            xmin = xx[0]
            if xx[-1] < 7:
                xmax = xx[-1]
                            
            nticks = 0.2
            xmin = dist[0]
            nticks = int((xmax-xmin)/5)
            if (xmax - xmin) < 10:
                nticks = 1
            if (xmax - xmin) < 5.5:
                nticks = 0.5
            if (xmax - xmin) < 3.0:
                nticks = 0.4
            if (xmax - xmin) < 1.5:
                nticks = 0.2  
            if (xmax - xmin) < 0.5:
                nticks = 0.1  
            if (xmax - xmin) < 0.1:
                nticks = 0.05     
            if (xmax - xmin) < 0.05:
                nticks = 0.02    
            if (xmax - xmin) < 0.03:
                nticks = 0.01
            if (xmax - xmin) < 0.01:
                nticks = 0.005               
            
            nro = 1
            if nticks < 0.1:
                nro = 2
            if nticks < 0.01:
                nro = 3

            major_yticks = [np.around(a,nro) for a in np.arange(xmin, xmax, nticks)]

            major_yticks[0] -= np.around(nticks/2,nro)
            xmin = major_yticks[0]
            mm = []
            for nn in range(100):
                val = nn*nticks+major_yticks[0]

                if val < 7:
                    mm.append(np.around(val,nro))
                else:
                    break
            
            if 'water+mol' in nm and nn2 == 1:
                dist = np.arange(dist.shape[0],dtype=int)+1
                xlabel = "# Points (not CM-CM distance)"         
                xmin = 0.5
                xmax = dist[-1]+0.5
                mm = list(range(dist[-1]+1))

            for ii in range(3):
                ylim = ax[kk][ii].get_ylim()
                yy = ax[kk][ii].get_yticks()

                if ii == 2:
                    ax[kk][ii].set_yticks(yy[:-1])
                ax[kk][ii].set_xticks(mm[1:])

                ax[kk][ii].axis([xmin,xmax,yy[0],yy[-1]])

                ax[kk][ii].yaxis.set_minor_locator(AutoMinorLocator(n=1))
                        
                ax[kk][ii].grid(which='both')
            
            
            if kk == nn2-1:
                ax[kk][0].set_xlabel(xlabel,fontsize=25)
                ax[kk][1].set_xlabel(xlabel,fontsize=25)
                ax[kk][2].set_xlabel(xlabel,fontsize=25)
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Total',**marker_style2)    
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='CCSD(T)/SAPT Total',**marker_style1)

                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Electrostatics',**marker_style4)
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT Electrostatics',**marker_style3)
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Dispersion',**marker_style8)
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT Dispersion',**marker_style7)

                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Induction',**marker_style10)
                ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT Induction',**marker_style9)

                ylim = ax[kk][1].get_ylim()
                yy = ax[kk][1].get_yticks()
                ax[kk][1].axis([xmin,xmax,yy[0],yy[-1]])
                ax[kk][1].legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(1.5, -.30),
                ncol=4, fancybox=True, shadow=True)
        
        plt.subplots_adjust(wspace=0.15,hspace=0.2)
        plt.grid(True)
        if save:            
            plt.savefig(f'{self.resdir}/sapt_dimers.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')
        if show:
            plt.show()

        plt.close()


    def make_sapt_plot_desres(self,save=True,show=False,todos=False,maxp=7):
        if not self.molfit.do_sapt_dimers:
            return
        
        nsort = np.argsort(self.molfit.sapt_dimers)  
        plotnm = []
        for i,kk in enumerate(nsort):
            nm = self.molfit.sapt_dimers[kk]
            if 'DES' not in nm:
                continue
            if 'DES' in nm and 'sapt' in nm:
                continue
            plotnm.append(nm)

        if len(plotnm) == 0:
            return

        marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o',ms=6)
        marker_style2 = dict(color='black', linestyle='--',linewidth=2,marker='o',ms=4)

        marker_style3 = dict(color='red', linestyle='-',linewidth=2,marker='v',ms=6)
        marker_style4 = dict(color='red', linestyle='--',linewidth=2,marker='v',ms=4)

        marker_style5 = dict(color='blue', linestyle='-',linewidth=2,marker='D',ms=6)
        marker_style6 = dict(color='blue', linestyle='--',linewidth=2,marker='D',ms=4)

        marker_style7 = dict(color='green', linestyle='-',linewidth=2,marker='x',ms=6)
        marker_style8 = dict(color='green', linestyle='--',linewidth=2,marker='x',ms=4)

        marker_style9 = dict(color='magenta', linestyle='-',linewidth=2,marker='s',ms=6)
        marker_style10 = dict(color='magenta', linestyle='--',linewidth=2,marker='s',ms=4)

        compnts = ['electrostatics','exchange','induction','dispersion','total']
        xlabel = r"CM-CM distance (${\AA}$)"

        nplots = 0

        results = []
        nall = len(self.molfit.sapt_dimers)
        
        for i,nm in enumerate(plotnm):
            if 'water' in nm:
                cid2 = watercid
            else:
                cid2 = self.cid
            
            ranges = []
            nstart = 0
            for gid,dt in desres_dimer_info[self.cid][cid2].items():
                if dt[0] == 'md_dimer' or dt[0] == 'qm_opt_dimer':
                    ranges.append([nstart,nstart+dt[1]])
                nstart+=dt[1]
            nplots = 2*len(ranges)
            if nplots == 0:
                continue

            n1 = len(ranges)
                        
            res = self.ccsdt_dimer[nm]
            res = self.sapt_dimer[nm]
                
            sapt_ref = self.molfit.sapt_dimers_ref[nm]
            ccsdt_ref = self.molfit.ccsdt_dimers_ref[nm]
                
            natm = process.count_atoms(f"{self.path}/liquid/gas.xyz")
            if 'water' in nm:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,3])
            else:
                cm1,cm2,cm = process.compute_center_of_mass(f"{self.path}/dimer/{nm}.arc",[natm,natm])
            dist = np.linalg.norm(cm2-cm1,axis=1)
            dist = dist[self.molfit.ccsdt_dimers_indx[nm]]
            
            sizes = [8,16,32,36,38,40]
    #         n1 = int(np.ceil(nplots/2))
            
            if n1-1 > len(sizes):
                df = sizes[-1]
                for j in range(n1-len(sizes)):
                    sizes += [df+8*(j+1)]         
            
            if todos:
                maxp = n1
            nn1 = np.arange(n1)
            if n1 > maxp:
                nn2 = maxp
            else:
                nn2 = n1

            if n1 > 0:
                f, ax = plt.subplots(nn2,3,figsize=(36, sizes[nn2-1]))
            else:
                return
            
            if nn2 == 1:
                ax = [ax]
    #         for kk in range(nn1):
            cpl = 0
            for kk,pp in enumerate(nn1):
                if cpl == maxp:
                    break
                n0,nf = ranges[pp]

                m1 = self.molfit.sapt_dimers_indx[nm] > n0-1
                m2 = self.molfit.sapt_dimers_indx[nm] < nf
                m4 = m1&m2
                m5 = dist < 7.2
                m3 = m4&m5
                indx = self.molfit.sapt_dimers_indx[nm][m3]

                p0 = 1
                pff = indx.shape[0]+1
                dist2 = dist[m3]
                if dist2.shape[0] < 5:
                    continue
                ax[kk][0].plot(dist[m3],res[m3][:,-1],**marker_style2)    
                ax[kk][0].plot(dist[m3],ccsdt_ref[m3],**marker_style1,label=f"CCSD(T) Total - {indx[0]}-{indx[-1]}")
                ax[kk][1].plot(dist[m3],res[m3][:,0],**marker_style4)
                ax[kk][1].plot(dist[m3],sapt_ref[m3][:,0],**marker_style3)
                ax[kk][1].plot(dist[m3],res[m3][:,3],**marker_style8)
                ax[kk][1].plot(dist[m3],sapt_ref[m3][:,3],**marker_style7)
                ax[kk][2].plot(dist[m3],res[m3][:,2],**marker_style10)
                ax[kk][2].plot(dist[m3],sapt_ref[m3][:,2],**marker_style9)

                db = nm.split('_')[0]
                if "water" in nm:
                        title = f"Water + #{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"
                else:
                    title = f"#{self.n} {database_full[self.n][0].capitalize():<s} Dimer - {db} conformations"

                if kk == 0:
                    ax[kk][2].set_title(title,fontsize=25,loc='right')


                ax[kk][0].set_ylabel('Energy (kcal/mol)',fontsize=25)
        #         ax[kk][1].set_title(nm,fontsize=22)
                lims = np.zeros((3,4))
                
                xlim = ax[kk][0].get_xlim()
                xx = ax[kk][0].get_xticks()
                xmax = 7
                xmin = xx[0]
                if xx[-1] < 7:
                    xmax = xx[-1]
                                
                nticks = 0.2
                xmin = dist2[0]
                nticks = int((xmax-xmin)/5)
                if (xmax - xmin) < 10:
                    nticks = 1
                if (xmax - xmin) < 5.5:
                    nticks = 0.5
                if (xmax - xmin) < 3.0:
                    nticks = 0.4
                if (xmax - xmin) < 1.5:
                    nticks = 0.2  
                if (xmax - xmin) < 0.5:
                    nticks = 0.1                        
                
                major_yticks = [np.around(a,1) for a in np.arange(xmin, xmax, nticks)]
                major_yticks[0] -= np.around(nticks/2,1)
                xmin = major_yticks[0]
                mm = []
                for nn in range(100):
                    val = nn*nticks+major_yticks[0]

                    if val < 7:
                        mm.append(np.around(val,1))
                    else:
                        break

                for ii in range(3):
                    ylim = ax[kk][ii].get_ylim()
                    yy = ax[kk][ii].get_yticks()
                    ax[kk][ii].set_xticks(mm[1:])

                    if ii == 2:
                        ax[kk][ii].set_yticks(yy[:-1])
                    ax[kk][ii].axis([xmin,xmax,yy[0],yy[-1]])

                    ax[kk][ii].yaxis.set_minor_locator(AutoMinorLocator(n=1))
                    
                                    
                    ax[kk][ii].grid(which='both')            

                if kk == nn2-1:
                    ax[kk][0].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=25)
                    ax[kk][1].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=25)
                    ax[kk][2].set_xlabel(r"CM-CM distance (${\AA}$)",fontsize=25)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Total',**marker_style2)    
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='CCSD(T)/CBS Total',**marker_style1)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Electrostatics',**marker_style4)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT0 Electrostatics',**marker_style3)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Dispersion',**marker_style8)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT0 Dispersion',**marker_style7)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='HIPPO Induction',**marker_style10)
                    ax[kk][1].plot(xlim[0]-2,ylim[0]-2,label='SAPT0 Induction',**marker_style9)

                    ylim = ax[kk][1].get_ylim()
                    yy = ax[kk][1].get_yticks()
                    ax[kk][1].axis([xmin,xmax,yy[0],yy[-1]])
                    ax[kk][1].legend(frameon=True,fontsize=22,loc='center right', bbox_to_anchor=(1.45, -.35),
                    ncol=4, fancybox=True, shadow=True)

                cpl += 1

            plt.subplots_adjust(wspace=0.15,hspace=0.15)
            plt.grid(True)

            if save:
                plt.savefig(f'{self.resdir}/{nm}-sapt.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')

            if show:
                plt.show()


            plt.close()

    def make_opt_sapt_plots(self,save=True,show=False):
        if not self.molfit.do_dimers:
            return
        
        xdata = np.array([1,2,3,4,5])
        xlabel = "Points"

        marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o',ms=8)
        marker_style2 = dict(color='black', linestyle='--',linewidth=2,marker='o',ms=8)

        marker_style3 = dict(color='red', linestyle='-',linewidth=2,marker='v',ms=8)
        marker_style4 = dict(color='red', linestyle='--',linewidth=2,marker='v',ms=8)

        marker_style5 = dict(color='blue', linestyle='-',linewidth=2,marker='D',ms=8)
        marker_style6 = dict(color='blue', linestyle='--',linewidth=2,marker='D',ms=8)

        marker_style7 = dict(color='green', linestyle='-',linewidth=2,marker='x',ms=8)
        marker_style8 = dict(color='green', linestyle='--',linewidth=2,marker='x',ms=8)

        marker_style9 = dict(color='magenta', linestyle='-',linewidth=2,marker='s',ms=8)
        marker_style10 = dict(color='magenta', linestyle='--',linewidth=2,marker='s',ms=8)

        compnts = ['electrostatics','exchange','induction','dispersion','total']

        nm = 'water+mol'
        sapt_ref = self.molfit.ref_energy[nm]
        res = self.opt_dimer
        
        dist = np.zeros(5)
        for k in range(5):
            natm = process.count_atoms(f"{self.path}/liquid/gas.xyz")
            cm,cm1,cm2 = process.compute_center_of_mass(f"{self.path}/dimer/water+mol-conf_{k+1}.xyz",[natm,3])
            d = np.linalg.norm(cm2-cm1,axis=1)
            dist[k] += d[0]
        indx = np.argsort(dist)
        dist = dist[indx]
        res = res[indx]
        sapt_ref = sapt_ref[indx]

        plt.rcParams.update({'font.size': 18}) 
        f, ax = plt.subplots(figsize=(12, 8))
        
        indx = sapt_ref[:,-1] < 10
        sapt_ref=sapt_ref[indx]
        res = res[indx]
        npt = sapt_ref.shape[0]
        p0 = 0
        pf = npt
        ax.plot(xdata[p0:pf],res[p0:pf,-1],label='HIPPO Total',**marker_style2)    
        ax.plot(xdata[p0:pf],res[p0:pf,0],label='HIPPO Electrostatics',**marker_style4)
        ax.plot(xdata[p0:pf],res[p0:pf,3],label='HIPPO Dispersion',**marker_style8)
        ax.plot(xdata[p0:pf],res[p0:pf,2],label='HIPPO Induction',**marker_style10)

        ax.plot(xdata[p0:pf],sapt_ref[p0:pf,-1],label='SAPT Total',**marker_style1)
        ax.plot(xdata[p0:pf],sapt_ref[p0:pf,0],label='SAPT Electrostatics',**marker_style3)
        ax.plot(xdata[p0:pf],sapt_ref[p0:pf,3],label='SAPT Dispersion',**marker_style7)
        ax.plot(xdata[p0:pf],sapt_ref[p0:pf,2],label='SAPT Induction',**marker_style9)

        plt.xlabel(xlabel,fontsize=25)
        plt.ylabel('Energy (kcal/mol)',fontsize=25)

        ax.title.set_text(nm)

        ax.set_xticks(xdata[p0:pf])
        plt.minorticks_off()
        ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(n=1))
        ax.grid(which='both')

        plt.legend(loc=0)
        ax.legend(frameon=True,fontsize=22,loc='lower center', bbox_to_anchor=(0.5, -.5),
            ncol=2, fancybox=True, shadow=True)

        if save:
            plt.savefig(f'{self.resdir}/water+mol_opt.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')

        if show:
            plt.show()

        plt.close()

    
    def benzene_parallel_disp(self,save=True,show=False):
        if self.n != 148:
            return

        n = self.n
        r1list = [3.2,3.4,3.6,3.8]

        colors = ['blue','magenta','red','dodgerblue']

        nm = "parallel-disp-ccsdt"
        ### Reference energy
        df = pd.read_csv(f"{datadir}/qm-calc/{n}/ccsdt_dimers/{nm}-energy.csv", sep=',',)
        ref = df.to_numpy()
        xdata1 = ref[:,1][:21]

        f, ax = plt.subplots(figsize=(12, 8))

        linestyles = ['']

        for p,i in enumerate(range(0,84,21)):    
            sapt_ref = self.molfit.ccsdt_dimers_ref[nm]
            res = self.ccsdt_dimer[nm]
            
            res = res[i:i+21]
            sapt_ref = sapt_ref[i:i+21]

        #     err = (res-sapt_ref)**2
        #     errors.append(err)

        #     if p == 2 or p == 3:
        #         continue
            if p == 0 or p == 1:
                continue
            
            
            #f, ax = plt.subplots()

            #ax.set(xlim=(0, 50), ylim=(30, 70))
        #     marker_style1 = dict(color='black', linestyle='-',linewidth=2,marker='o')
        #     marker_style2 = dict(color='green', linestyle='--', marker='s')
        #     marker_style3 = dict(color='red', linestyle='dotted', marker='v')
            
            marker_style1 = dict(color=colors[p], linestyle='-',linewidth=2,marker='o')
            marker_style2 = dict(color=colors[p], linestyle='--', marker='s')
            marker_style3 = dict(color=colors[p], linestyle='dotted', marker='v')

        #     ax.plot(xdata1,sapt_ref,label="CCSD(T) Total",**marker_style1)
            ax.plot(xdata1,res,label="HIPPO Total",**marker_style2)
            ax.plot(xdata1,sapt_ref,label=f"CCSD(T) R1 = {r1list[p]}",**marker_style1)
        #     ax.plot(xdata1,res,label="HIPPO",**marker_style2)
        #     ax.plot(xdata1,res2,label="Amoeba",**marker_style3)
                
            plt.xlabel("R2 (A)",fontsize=25)
            plt.ylabel('Energy (kcal/mol)',fontsize=25)
            
            ax.title.set_text(nm)
            
        #     if p != 0:
        #         ax.axis([-2.7,1.3,-2.7,1.3])

            ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
            ax.grid(which='both')
            
        #     if k == 0:
        #         ax.axis([0.65,1.15,-5,10])
            #ax.plot(ax.get_xlim(), ax.get_ylim(), linewidth=1, ls="--", c=".3")

            plt.legend(loc=0)
            ax.legend(frameon=True,fontsize=22,loc='upper right', bbox_to_anchor=(1.5, 0.8),
                ncol=1, fancybox=True, shadow=True)

            #plt.savefig('dimer_radial.eps',format='eps')

        plt.grid(True)

        if save:
            plt.savefig(f'{self.resdir}/{nm}.pdf', format='pdf', dpi=1000,transparent=True, bbox_inches='tight')
        if show:
            plt.show()

        plt.close()