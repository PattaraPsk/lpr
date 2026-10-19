"""
Unit and Integration tests for Thai LPR Annotator:
- Subfolder scanning & grouping
- Single image delete
- Custom rename
- Batch removal of '_dup' duplicate images
- GUI instantiation with 9-column Check View
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app.session_manager import SessionManager, ImageItem
from app.provinces import match_province_value, get_province_entries
from app.plate_types import get_plate_type_entries, match_plate_type, PLATE_TYPES_DATA

def test_provinces_and_plate_types():
    print("[1/6] Testing Provinces & Plate Types...")
    prov_entries = get_province_entries()
    assert len(prov_entries) >= 79
    
    pt_entries = get_plate_type_entries()
    assert len(pt_entries) == 12
    
    assert match_plate_type("WhiteBlackCar") == "WhiteBlackCar"
    assert match_plate_type("white_black_car") == "WhiteBlackCar"
    assert match_plate_type("WhiteBlackMotor") == "WhiteBlackMotor"
    assert match_plate_type("white_black_motor") == "WhiteBlackMotor"
    assert match_plate_type("WhiteBlue") == "WhiteBlue"
    assert match_plate_type("YellowBlack") == "YellowBlack"
    assert match_plate_type("+") == "+"
    assert match_plate_type("-") == "-"
    print("  -> Provinces & Plate Types OK")

def test_subfolders_and_parsing():
    print("[2/6] Testing Subfolder Loading & Grouping...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "cam1").mkdir()
        (tmp_path / "cam2").mkdir()
        
        # Root file
        (tmp_path / "root_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").touch()
        # Cam1 files
        (tmp_path / "cam1" / "img01_WhiteBlue_2ขข_2222_เชียงใหม่.jpg").touch()
        (tmp_path / "cam1" / "img01_dup_WhiteBlue_2ขข_2222_เชียงใหม่.jpg").touch()
        # Cam2 files
        (tmp_path / "cam2" / "img02_YellowBlack_3คค_3333_ภูเก็ต.jpg").touch()
        (tmp_path / "cam2" / "img03_+_+_+_+.jpg").touch()
        (tmp_path / "cam2" / "img04_-_-_-_-.jpg").touch()
        
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        assert len(session.items) == 6
        
        # Test subfolders detected
        subs = session.get_subfolders()
        assert subs == ["cam1", "cam2"], f"Expected ['cam1', 'cam2'], got {subs}"
        
        # Test subfolder filtering
        cam1_indices = session.get_filtered_indices("cam1")
        assert len(cam1_indices) == 2
        
        cam2_indices = session.get_filtered_indices("cam2")
        assert len(cam2_indices) == 3
        
        root_indices = session.get_filtered_indices("(Root)")
        assert len(root_indices) == 1
        
        # Test navigation within filter
        next_cam1 = session.get_next_index(cam1_indices[0], "cam1")
        assert next_cam1 == cam1_indices[1]
        
        prev_cam1 = session.get_prev_index(cam1_indices[1], "cam1")
        assert prev_cam1 == cam1_indices[0]
        
        print("  -> Subfolders & Grouping OK")

def test_delete_and_custom_rename():
    print("[3/6] Testing Delete & Custom Rename...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_file = tmp_path / "car01_WhiteBlackCar_1กก_1234_กรุงเทพมหานคร.jpg"
        img_file.touch()
        
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        assert len(session.items) == 1
        
        # Custom Rename
        assert session.custom_rename(0, "custom_car_WhiteBlackCar_9กข_8888_ชลบุรี.jpg")
        assert session.items[0].filename == "custom_car_WhiteBlackCar_9กข_8888_ชลบุรี.jpg"
        assert session.items[0].d3 == "9กข"
        assert session.items[0].d4 == "8888"
        assert session.items[0].province == "ชลบุรี"
        assert (tmp_path / "custom_car_WhiteBlackCar_9กข_8888_ชลบุรี.jpg").exists()
        
        # Delete item
        assert session.delete_item(0)
        assert len(session.items) == 0
        assert not (tmp_path / "custom_car_WhiteBlackCar_9กข_8888_ชลบุรี.jpg").exists()
        print("  -> Delete & Custom Rename OK")

def test_remove_dup_images():
    print("[4/6] Testing Remove _dup Duplicate Images...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "normal1_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").touch()
        (tmp_path / "normal1_dup_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").touch()
        (tmp_path / "normal1_dup1_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").touch()
        (tmp_path / "normal2_WhiteBlue_2ขข_2222_เชียงใหม่.jpg").touch()
        
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        assert len(session.items) == 4
        
        dup_indices = session.find_dup_items()
        assert len(dup_indices) == 2
        
        removed = session.remove_dup_images()
        assert removed == 2
        assert len(session.items) == 2
        assert not (tmp_path / "normal1_dup_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").exists()
        assert not (tmp_path / "normal1_dup1_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").exists()
        assert (tmp_path / "normal1_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg").exists()
        print("  -> Remove _dup Duplicate Images OK")

def test_session_persistence():
    print("[5/6] Testing Session Persistence across Subfolders...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "sub1").mkdir()
        img = tmp_path / "sub1" / "car_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg"
        img.touch()
        
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        session.approve_and_save(0, "car", "WhiteGreen", "5มม", "7777", "เชียงใหม่")
        
        # Reload
        session2 = SessionManager()
        assert session2.load_folder(str(tmp_path))
        item = session2.items[0]
        assert item.status == "approved"
        assert item.plate_type == "WhiteGreen"
        assert item.subfolder == "sub1"
        assert item.rel_path == "sub1/car_WhiteGreen_5มม_7777_เชียงใหม่.jpg"
        print("  -> Session Persistence OK")

def test_gui_headless():
    print("[6/6] Testing GUI components with Subfolders & 9-column Check View...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from app.main_window import MainWindow
    from app.check_view import CheckViewDialog
    
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    
    sample_dir = str((Path(__file__).parent / "sample_images").resolve())
    assert window.session.load_folder(sample_dir)
    assert len(window.session.items) >= 10
    
    window._show_item_at(0)
    assert window.txt_name.text() != ""
    assert window.combo_plate_type.get_plate_type_value() != ""
    
    # Check dialog test
    dlg = CheckViewDialog(window.session, 0, window)
    assert dlg.table.columnCount() == 9 # #, Status, Subfolder, Filename, Name, Plate Type, d3, d4, Province
    assert dlg.table.rowCount() == len(window.session.items)
    
    # Test Not Plate fill behavior (does not advance, only fills)
    window._on_not_plate_clicked()
    assert window.session.current_index == 0
    assert window.combo_plate_type.get_plate_type_value() == "+"
    assert window.txt_d3.text() == "+"
    assert window.txt_d4.text() == "+"
    assert window.combo_province.get_province_value() == "+"
    
    # Test Cannot Read fill behavior (does not advance, only fills)
    window._on_cannot_read_clicked()
    assert window.session.current_index == 0
    assert window.combo_plate_type.get_plate_type_value() == "-"
    assert window.txt_d3.text() == "-"
    assert window.txt_d4.text() == "-"
    assert window.combo_province.get_province_value() == "-"
    
    print("  -> GUI Headless Instantiation & Load OK")


if __name__ == "__main__":
    test_provinces_and_plate_types()
    test_subfolders_and_parsing()
    test_delete_and_custom_rename()
    test_remove_dup_images()
    test_session_persistence()
    test_gui_headless()
    print("\nALL TESTS PASSED!")
