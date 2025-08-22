"""pixel_art_editor.py – PyQt5 Pixel‑Art Animation Editor

Features
--------
* Editable grid of squares (default 8×8, click to toggle white/black)
* Unlimited frames with thumbnail preview strip (click thumbnail to select)
* Add/Delete frame, Play/Stop animation (QTimer)
* Dynamic grid resize at any time – arrays are padded/cropped as requested
* Export / Import full animation as 3‑D ``.npy`` or current frame as 2‑D ``.npy``

Dependencies
------------
::

   pip install PyQt5 numpy

Run with ``python pixel_art_editor.py``.
"""

from __future__ import annotations

import sys
import typing as _t
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

CELL_SIZE = 30  # pixels per cell in main grid
THUMB_SCALE = 40  # pixels per cell in thumbnail preview (进一步增大预览图)


class PixelArtEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Art Animation Editor (PyQt5)")

        # 设置窗口为较大的正方形
        self.setGeometry(100, 100, 1000, 1000)
        self.setMinimumSize(600, 600)

        # ------- Data model -------
        self.rows = 8
        self.cols = 8
        self.frames: list[np.ndarray] = [
            np.zeros((self.rows, self.cols), dtype=np.uint8)
        ]
        self.current_index = 0
        self.playing = False
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.next_frame)

        # 添加鼠标拖动状态
        self.is_dragging = False
        self.mouse_pressed = False
        self.start_pos = None
        self.drag_value = 0  # 拖拽时设置的目标值

        # ------- UI -------
        central = QWidget(self)
        self.setCentralWidget(central)
        main_v = QVBoxLayout(central)

        # Thumbnail preview list
        self.thumb_list = QListWidget()
        self.thumb_list.setViewMode(QListWidget.IconMode)
        self.thumb_list.setFlow(QListWidget.LeftToRight)
        self.thumb_list.setIconSize(QPixmap(THUMB_SCALE, THUMB_SCALE).size())
        self.thumb_list.setFixedHeight(self.rows * THUMB_SCALE + 40)
        self.thumb_list.itemClicked.connect(self.on_thumb_clicked)

        main_v.addWidget(self.thumb_list)

        # Grid table
        self.table = QTableWidget(self.rows, self.cols)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)

        # 鼠标事件处理 - 简化的事件连接
        self.table.cellPressed.connect(self.on_cell_pressed)
        self.table.cellEntered.connect(self.on_cell_entered)

        # 确保表格能接收鼠标事件
        self.table.setMouseTracking(False)  # 初始状态关闭鼠标追踪

        # 重写表格的鼠标释放事件
        self.table.mouseReleaseEvent = self.table_mouse_release_event

        main_v.addWidget(self.table)

        # Toolbar buttons - 重新设计布局
        # 上方工具栏：文件操作和画布控制
        toolbar_top = QToolBar()
        self.addToolBar(Qt.TopToolBarArea, toolbar_top)
        toolbar_top.setMovable(False)

        # 下方工具栏：帧操作和播放控制
        toolbar_bottom = QToolBar()
        self.addToolBar(Qt.BottomToolBarArea, toolbar_bottom)
        toolbar_bottom.setMovable(False)

        btn_add = QPushButton("Add Frame")
        btn_copy = QPushButton("Copy Frame")
        btn_del = QPushButton("Delete Frame")
        btn_play = QPushButton("Play")
        btn_stop = QPushButton("Stop")
        btn_resize = QPushButton("Resize Grid")
        btn_exp_all = QPushButton("Export All")
        btn_exp_fr = QPushButton("Export Frame")
        btn_load = QPushButton("Load")

        btn_add.clicked.connect(self.add_frame)
        btn_copy.clicked.connect(self.copy_frame)
        btn_del.clicked.connect(self.delete_frame)
        btn_play.clicked.connect(self.start_play)
        btn_stop.clicked.connect(self.stop_play)
        btn_resize.clicked.connect(self.resize_grid_dialog)
        btn_exp_all.clicked.connect(self.export_all)
        btn_exp_fr.clicked.connect(self.export_frame)
        btn_load.clicked.connect(self.load_frames)

        # 上方工具栏：文件操作和画布控制
        for b in [btn_load, btn_exp_all, btn_exp_fr, btn_resize]:
            toolbar_top.addWidget(b)

        # 下方工具栏：帧操作和播放控制
        for b in [btn_add, btn_copy, btn_del, btn_play, btn_stop]:
            toolbar_bottom.addWidget(b)

        # Status label在下方工具栏
        toolbar_bottom.addSeparator()
        self.status_lbl = QLabel()
        toolbar_bottom.addWidget(self.status_lbl)

        self.update_table_geometry()
        self.refresh_thumbs()
        self.update_status()

    # ---------- Helpers ----------
    def current_frame(self) -> np.ndarray:
        return self.frames[self.current_index]

    def toggle_cell(self, row: int, col: int):
        """反色单元格状态"""
        if self.playing:
            return
        frame = self.current_frame()
        frame[row, col] ^= 1  # 反色操作：0变1，1变0
        self.update_cell_display(row, col)
        self.refresh_thumb(self.current_index)

    def set_cell(self, row: int, col: int, value: int):
        """设置单元格为指定值"""
        if self.playing:
            return
        frame = self.current_frame()
        if frame[row, col] != value:  # 只有在值不同时才更新
            frame[row, col] = value
            self.update_cell_display(row, col)
            self.refresh_thumb(self.current_index)

    def on_cell_pressed(self, row: int, col: int):
        """鼠标按下时的处理"""
        if self.playing:
            return

        self.mouse_pressed = True
        self.start_pos = (row, col)
        self.is_dragging = False

        # 反色按下的单元格
        self.toggle_cell(row, col)

        # 启用鼠标追踪以便检测拖动
        self.table.setMouseTracking(True)

    def on_cell_entered(self, row: int, col: int):
        """鼠标进入单元格时的处理 - 拖动时反色每个经过的像素"""
        if self.playing or not self.mouse_pressed:
            return

        current_pos = (row, col)

        # 如果从起始位置移动了，就开始拖拽模式
        if self.start_pos and current_pos != self.start_pos:
            self.is_dragging = True

        # 在拖拽模式下，反色当前单元格
        if self.is_dragging:
            self.toggle_cell(row, col)

    def table_mouse_release_event(self, event):
        """表格的鼠标释放事件处理"""
        if hasattr(self, "mouse_pressed") and self.mouse_pressed:
            self.mouse_pressed = False
            self.is_dragging = False
            self.start_pos = None
            # 关闭鼠标追踪以提高性能
            self.table.setMouseTracking(False)
        # 调用原始的鼠标释放事件
        QTableWidget.mouseReleaseEvent(self.table, event)

    def update_cell_display(self, row: int, col: int):
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, col, item)
        color = QColor("black") if self.current_frame()[row, col] else QColor("white")
        item.setBackground(color)

    def update_table_geometry(self):
        self.table.setRowCount(self.rows)
        self.table.setColumnCount(self.cols)
        for r in range(self.rows):
            self.table.setRowHeight(r, CELL_SIZE)
        for c in range(self.cols):
            self.table.setColumnWidth(c, CELL_SIZE)
        # adjust thumbnail list height
        self.thumb_list.setFixedHeight(self.rows * THUMB_SCALE + 40)
        self.redraw_current_frame()

    def redraw_current_frame(self):
        self.table.clear()
        for r in range(self.rows):
            for c in range(self.cols):
                self.update_cell_display(r, c)

    # ---------- Frame thumbnails ----------
    def frame_to_pixmap(self, arr: np.ndarray) -> QPixmap:
        h, w = arr.shape
        img = QImage(w, h, QImage.Format_RGB32)
        white = QColor("white").rgb()
        black = QColor("black").rgb()
        for y in range(h):
            for x in range(w):
                img.setPixel(x, y, black if arr[y, x] else white)
        return QPixmap.fromImage(img.scaled(w * THUMB_SCALE, h * THUMB_SCALE))

    def refresh_thumbs(self):
        self.thumb_list.clear()
        for idx, fr in enumerate(self.frames):
            item = QListWidgetItem()
            item.setIcon(QIcon(self.frame_to_pixmap(fr)))
            item.setText(str(idx + 1))
            self.thumb_list.addItem(item)
        self.thumb_list.setCurrentRow(self.current_index)

    def refresh_thumb(self, idx: int):
        item = self.thumb_list.item(idx)
        if item:
            item.setIcon(QIcon(self.frame_to_pixmap(self.frames[idx])))

    # ---------- Frame operations ----------
    def on_thumb_clicked(self, item: QListWidgetItem):
        idx = self.thumb_list.row(item)
        self.current_index = idx
        self.redraw_current_frame()
        self.update_status()

    def add_frame(self):
        self.frames.insert(
            self.current_index + 1, np.zeros((self.rows, self.cols), dtype=np.uint8)
        )
        self.current_index += 1
        self.refresh_thumbs()
        self.update_status()
        self.redraw_current_frame()

    def copy_frame(self):
        """复制当前帧并插入到下一个位置"""
        current_frame_copy = self.current_frame().copy()
        self.frames.insert(self.current_index + 1, current_frame_copy)
        self.current_index += 1
        self.refresh_thumbs()
        self.update_status()
        self.redraw_current_frame()

    def delete_frame(self):
        if len(self.frames) == 1:
            QMessageBox.information(self, "Info", "Cannot delete the only frame.")
            return
        self.frames.pop(self.current_index)
        self.current_index = max(0, self.current_index - 1)
        self.refresh_thumbs()
        self.update_status()
        self.redraw_current_frame()

    def next_frame(self):
        self.current_index = (self.current_index + 1) % len(self.frames)
        self.thumb_list.setCurrentRow(self.current_index)
        self.redraw_current_frame()
        self.update_status()

    # ---------- Playback ----------
    def start_play(self):
        if not self.playing:
            self.playing = True
            self.timer.start()

    def stop_play(self):
        self.playing = False
        self.timer.stop()

    # ---------- Resize Grid ----------
    def resize_grid_dialog(self):
        new_rows, ok1 = QInputDialog.getInt(
            self, "Resize", "Rows:", value=self.rows, min=1, max=64
        )
        if not ok1:
            return
        new_cols, ok2 = QInputDialog.getInt(
            self, "Resize", "Cols:", value=self.cols, min=1, max=64
        )
        if not ok2:
            return
        if (new_rows, new_cols) == (self.rows, self.cols):
            return
        self.apply_resize(new_rows, new_cols)

    def apply_resize(self, new_rows: int, new_cols: int):
        new_frames: list[np.ndarray] = []
        for fr in self.frames:
            h, w = fr.shape
            nr = np.zeros((new_rows, new_cols), dtype=np.uint8)
            min_r = min(h, new_rows)
            min_c = min(w, new_cols)
            nr[:min_r, :min_c] = fr[:min_r, :min_c]
            new_frames.append(nr)
        self.frames = new_frames
        self.rows, self.cols = new_rows, new_cols
        self.update_table_geometry()
        self.refresh_thumbs()
        self.update_status()

    # ---------- Export / Import ----------
    def export_all(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export animation", filter="NumPy Files (*.npy)"
        )
        if not path:
            return
        np.save(path, np.stack(self.frames, axis=0))
        QMessageBox.information(self, "Saved", f"Saved to {path}")

    def export_frame(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export frame", filter="NumPy Files (*.npy)"
        )
        if not path:
            return
        np.save(path, self.current_frame())
        QMessageBox.information(self, "Saved", f"Saved to {path}")

    def load_frames(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load animation", filter="NumPy Files (*.npy)"
        )
        if not path:
            return
        try:
            data = np.load(path)
            if data.ndim == 2:
                data = data[np.newaxis, ...]  # single frame
            if data.ndim != 3:
                raise ValueError("Expected 2‑D or 3‑D array")
            new_rows, new_cols = data.shape[1:]  # (frames, rows, cols)
            self.frames = [data[i].copy() for i in range(data.shape[0])]
            self.current_index = 0
            self.apply_resize(new_rows, new_cols)
            QMessageBox.information(self, "Loaded", f"Loaded {data.shape[0]} frame(s)")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ---------- Status ----------
    def update_status(self):
        self.status_lbl.setText(
            f"Frame {self.current_index + 1}/{len(self.frames)}  |  Grid {self.rows}×{self.cols}"
        )


# ---------- main entry ----------


def main():
    app = QApplication(sys.argv)
    w = PixelArtEditor()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
