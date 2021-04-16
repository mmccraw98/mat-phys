import os
import general as gmp
import viscofit as vf
import numpy as np
import matplotlib.pyplot as plt

forces = []
indentations = []
radii = []
times = []

print('loading data')
for folder in gmp.get_folders(os.path.join('data', 'S-Se_visco_201014')):
    for file in gmp.get_files(folder, req_ext='xlsx'):
        try:
            df = gmp.load(file)
        except:
            continue  # for some reason, there is an invisible file that causes an error here so we skip it

        zsens, defl, k, sampling_freq = df.Zsens, df.Defl, 3.6763, 2e3
        # cut everything before the maximum of the defl and then cut everything before the minimum of that new defl
        max = int(np.argmax(defl) * 0.9)
        min = np.argmin(defl[: max])

        zsens = zsens[min: max].values
        defl = defl[min: max].values

        # shift everything up to start at 0
        defl -= defl[0]
        zsens -= zsens[0]

        force = k * defl
        time = np.arange(0, force.size, 1) / sampling_freq
        indentation = zsens - defl
        radius = 1e-6

        plt.plot(indentation, force)
        plt.show()
        if gmp.selectyesno('keep? (y/n)'):
            forces.append(force), times.append(time), indentations.append(indentation), radii.append(radius)

maxwell_model = vf.maxwellModel(forces=forces, indentations=indentations, times=times, radii=radii)
voigt_model = vf.kelvinVoigtModel(forces=forces, indentations=indentations, times=times, radii=radii)
gmp.safesave(maxwell_model, os.path.join('data', 'S-Se_visco_201014', 'maxwell_model.pkl'))
gmp.safesave(maxwell_model, os.path.join('data', 'S-Se_visco_201014', 'voigt_model.pkl'))
quit()

print('performing fits')
results = maxwell_model.fit(maxiter=5000, max_model_size=4, num_attempts=100, save_global=True, fluidity=False)
gmp.safesave(results, os.path.join('data', 'S-Se_visco_201014', 'maxwell_fit.pkl'))
results = voigt_model.fit(maxiter=5000, max_model_size=4, num_attempts=100, save_global=True, fluidity=False)
gmp.safesave(results, os.path.join('data', 'S-Se_visco_201014', 'maxwell_fit.pkl'))
