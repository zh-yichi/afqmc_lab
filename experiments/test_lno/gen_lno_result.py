import numpy as np


def gen_lno_result(lno_thresh, nfrag, frag_list=None, out_file='lno_result.out'):
    """Rebuild ``lno_result.out`` from individual ``fragment.out{i+1}`` files.

    Parameters
    ----------
    lno_thresh : sequence of two floats
        The LNO thresholds, e.g. ``[1e-3, 1e-4]``.
    nfrag : int
        Total number of fragments.
    frag_list : list of int, optional
        Zero-based fragment indices to read. If ``None`` (default), uses
        ``range(nfrag)``. Fragment ``i`` is read from ``fragment.out{i+1}``.
    out_file : str
        Output file name (default ``lno_result.out``).
    """
    if frag_list is None:
        frag_list = list(range(nfrag))
    run_frag_list = list(frag_list)
    n = len(run_frag_list)

    las_center = [''] * n
    las_size = [None] * n
    lno_emp2 = np.zeros(n)
    lno_ecc = np.zeros(n)
    lno_eqmc = np.zeros(n)
    lno_eqmc_err = np.zeros(n)
    ccsd_time = np.zeros(n)
    qmc_time = np.zeros(n)

    for k, i in enumerate(run_frag_list):
        with open(f'fragment.out{i+1}', 'r') as f:
            for line in f:
                if 'LNO Center' in line:
                    las_center[k] = line.split()[-1]
                elif 'LNO-Active Space' in line and 'orbitals:' in line:
                    orb_part = line.split('orbitals:')[1]
                    orbs = [int(x) for x in orb_part.strip().strip('[]').replace(',', ' ').split()]
                    half = len(orbs) // 2
                    nactocc, nactvir = orbs[:half], orbs[half:]
                    las_size[k] = np.array([o + v for o, v in zip(nactocc, nactvir)], dtype=np.int32)
                elif 'LNO-MP2 Orbital Energy' in line:
                    lno_emp2[k] = float(line.split()[-1])
                elif 'LNO-CCSD Orbital Energy' in line:
                    lno_ecc[k] = float(line.split()[-1])
                elif 'LNO-AFQMC Orbital Energy' in line:
                    lno_eqmc[k] = float(line.split()[-3])
                    lno_eqmc_err[k] = float(line.split()[-1])
                elif 'LNO-CCSD Time' in line:
                    ccsd_time[k] = float(line.split()[-1])
                elif 'LNO-AFQMC Time' in line:
                    qmc_time[k] = float(line.split()[-1])

    las_size_arr = np.array(las_size, dtype=np.int32)
    las_max = las_size_arr.max()
    las_size = list(map(lambda row: f"{row}", las_size_arr))
    e_mp2 = np.sum(lno_emp2)
    e_ccsd = np.sum(lno_ecc)
    e_afqmc = np.sum(lno_eqmc)
    e_afqmc_err = np.sqrt(np.sum(lno_eqmc_err ** 2))
    tot_ccsd_time = np.sum(ccsd_time)
    tot_qmc_time = np.sum(qmc_time)

    with open(out_file, 'w') as f:
        width = 100
        f.write('=' * width + '\n')
        f.write(f'{"LNO-AFQMC Results":^{width}}\n')
        f.write('=' * width + '\n')

        f.write(f'{"Frag":>4s}  {"LAS Center":>14s}  {"LAS_SIZE":>8s}  '
                f'{"E(MP2)":>10s}  {"E(CCSD)":>10s}  '
                f'{"E(AFQMC)":>10s}  {"Error":>8s}  '
                f'{"t(CCSD)":>8s}  {"t(AFQMC)":>8s}\n')
        f.write('-' * width + '\n')

        for k, i in enumerate(run_frag_list):
            f.write(f"{i+1:4d}  {las_center[k]:>14s}  {las_size[k]:8s}  "
                    f"{lno_emp2[k]:10.8f}  {lno_ecc[k]:10.8f}  "
                    f"{lno_eqmc[k]:10.6f}  {lno_eqmc_err[k]:8.6f}  "
                    f"{ccsd_time[k]:8.2f}  {qmc_time[k]:8.2f}\n")

        f.write('-' * width + '\n')

        f.write(f'{"Summarize Fragments":^{width}}\n')
        f.write('-' * width + '\n')
        lno_thresh_str = "[" + ", ".join(f"{x:.2e}" for x in lno_thresh) + "]"
        f.write(f'{"LNO-Thresh":<20} {"Max LAS":>8} '
                f'{"E[MP2]":>12} {"E[CCSD]":>12} '
                f'{"E[AFQMC]":>10} {"Err[AFQMC]":>10} '
                f'{"CCSD-Time":>10} {"AFQMC-Time":>10}\n')

        f.write(f'{lno_thresh_str:<20} {las_max:>8} '
                f'{e_mp2:>12.8f} {e_ccsd:>12.8f} '
                f'{e_afqmc:>10.6f} {e_afqmc_err:>10.6f} '
                f'{tot_ccsd_time:>10.2f} {tot_qmc_time:>10.2f}\n')

        f.write('=' * width + '\n\n')

    return None


if __name__ == '__main__':
    gen_lno_result([1e-3, 1e-4], 10)
