# ============================================================================
# CELL KHỞI CHẠY (dán vào 1 notebook Colab mới, runtime CPU, rồi nhấn chạy)
# 1) Đăng nhập Drive  2) nhận file CS406-SCB-TumorSeg.zip (upload khi được hỏi,
# hoặc đã có sẵn trong /content)  3) giải nén vào MyDrive  4) chạy toàn bộ
#    notebooks/colab_preprocess_from_drive.ipynb: tải data -> Step 1-2-3 -> QC
# ============================================================================
import glob, os, zipfile
import nbformat
from google.colab import drive, files
from IPython import get_ipython

drive.mount('/content/drive')

zips = glob.glob('/content/CS406-SCB-TumorSeg*.zip')
if not zips:
    print('Chọn file CS406-SCB-TumorSeg.zip để upload...')
    files.upload()
    zips = glob.glob('/content/CS406-SCB-TumorSeg*.zip')
assert zips, 'Không thấy file zip'

with zipfile.ZipFile(zips[0]) as z:
    z.extractall('/content/drive/MyDrive')          # tạo MyDrive/CS406-SCB-TumorSeg/
print('Đã giải nén vào /content/drive/MyDrive/CS406-SCB-TumorSeg')

nb_path = '/content/drive/MyDrive/CS406-SCB-TumorSeg/notebooks/colab_preprocess_from_drive.ipynb'
nb = nbformat.read(nb_path, as_version=4)
ip = get_ipython()
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    print(f'\n===== Chạy cell {i} =====')
    res = ip.run_cell(cell.source)
    if not res.success:
        raise RuntimeError(f'Cell {i} lỗi - dừng lại, sửa rồi chạy lại cell khởi chạy này (tiến độ đã lưu, chạy tiếp được).')
