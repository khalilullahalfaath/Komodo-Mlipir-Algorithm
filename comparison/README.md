# Perbandingan MATLAB vs Python

Hasil: 30 run per fungsi (seed 1..30), F1-F23, dimensi 50 untuk F1-F13, 25000 evaluasi, MATLAB R2020a.
Tabel lengkap: `results/summary.txt` (dibuat oleh `python comparison/compare.py`).

## Temuan
- Fungsi benchmark: selisih relatif terbesar 5.8e-15 pada 110 titik (F7 dilewati karena mengandung `rand`).
- Python vs MATLAB + fix `AllHQ`: tidak beda signifikan, kecuali F8 (p=0.022, tidak lolos koreksi 23 uji).
- Python vs MATLAB asli: beda signifikan di F7, F8, F12, F13 (MATLAB asli lebih baik). Penyebabnya bug `AllHQ`.
- Jumlah evaluasi sama. F9: 2 dari 30 run MATLAB masuk tahap 2, Python tidak ada (variasi acak).
- Waktu per run: Python 2.4 s, MATLAB 3.0 s.

## Dua varian MATLAB
- `m_orig`: kode asli, hanya label Foxholes di `Evaluation.m` diubah dari `case 16` ke `case 14` (aslinya F14 error dan F16 jadi Foxholes).
- `m_fix`: `m_orig` + perbaikan `KMA2D.m`: `AllHQ(ind:ind+NumBM-1,:)` memakai `ind` yang melompat 5 baris padahal tiap swarm hanya 2 baris, sehingga `AllHQ` melebar dengan baris nol (titik asal ikut jadi penarik). Python memakai indeks yang benar.

## Cara mengulang
1. `python comparison/make_matlab_variants.py` (membuat `m_orig/` dan `m_fix/` dari `komodo_mlipir-1.0.1/`).
2. Di masing-masing folder: `matlab -batch run_kma` (hasil `kma_result.csv`, salin ke `results/matlab_orig.csv` / `matlab_fix.csv`).
3. `python comparison/run_py.py` (hasil `results/python.csv`).
4. Nilai fungsi: `python comparison/gen_points.py`, salin `results/points.csv` ke `m_orig/`, jalankan `matlab -batch eval_cmp`, hasilnya `eval_result.csv` -> `results/eval_points.csv`.
5. `python comparison/compare.py`.
