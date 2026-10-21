"""
Unit and Integration tests for Thai LPR Annotator:
- Subfolder scanning & grouping
- Single image delete
- Custom rename
- Batch removal of '_dup' duplicate images
- Session persistence
- GUI instantiation with 9-column Check View
- AI Reader (a1.3.4.1) inference, comparison, and selective/all replacement
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image

try:
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from app.session_manager import SessionManager, ImageItem
from app.provinces import match_province_value, get_province_entries
from app.plate_types import get_plate_type_entries, match_plate_type, PLATE_TYPES_DATA
from app.plate_reader import PlateReader, decode_tokens_to_plate


def test_provinces_and_plate_types():
    print("[1/7] Testing Provinces & Plate Types...")
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
    print("[2/7] Testing Subfolder Loading & Grouping...")
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
    print("[3/7] Testing Delete & Custom Rename...")
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
    print("[4/7] Testing Remove _dup Duplicate Images...")
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
    print("[5/7] Testing Session Persistence across Subfolders...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "sub1").mkdir()
        img = tmp_path / "sub1" / "car_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg"
        img.touch()
        
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        session.items[0].set_ai_prediction("1กก", "1111", "กรุงเทพมหานคร", 0.98)
        session.approve_and_save(0, "car", "WhiteGreen", "5มม", "7777", "เชียงใหม่")
        
        # Reload
        session2 = SessionManager()
        assert session2.load_folder(str(tmp_path))
        item = session2.items[0]
        assert item.status == "approved"
        assert item.plate_type == "WhiteGreen"
        assert item.subfolder == "sub1"
        assert item.rel_path == "sub1/car_WhiteGreen_5มม_7777_เชียงใหม่.jpg"
        assert item.ai_d3 == "1กก"
        assert item.ai_d4 == "1111"
        assert item.ai_province == "กรุงเทพมหานคร"
        assert item.ai_confidence == 0.98
        print("  -> Session Persistence OK")


def test_gui_headless():
    print("[6/7] Testing GUI components with Subfolders & 9-column Check View...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from app.main_window import MainWindow
    from app.check_view import CheckViewDialog
    
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "cam1").mkdir()
        for i in range(10):
            # Create a small valid mock JPEG image
            img_file = tmp_path / "cam1" / f"car{i:02d}_WhiteBlackCar_1กก_{1000+i}_กรุงเทพมหานคร.jpg"
            img = Image.new("RGB", (100, 32), color=(255, 255, 255))
            img.save(img_file)
            
        sample_dir = str(tmp_path)
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
        
        window._stop_workers()
        
    print("  -> GUI Headless Instantiation & Load OK")


def test_ai_reader_comparison_and_replace():
    print("[7/7] Testing AI Reader (a1.3.4.1) Inference, Comparison & Replacement...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from app.main_window import MainWindow

    # 1. Test PlateReader instance and prediction
    reader = PlateReader.get_instance()
    if reader.weights_path.is_file() and reader.model is not None:
        sample_plate = Path("dataset/curated_lpr_dataset/plates/AA15_car_ศส_1009_กรุงเทพมหานคร.jpg")
        if sample_plate.exists():
            res = reader.predict(sample_plate)
            assert res["error"] is None, f"Prediction error: {res['error']}"
            assert res["d3"] == "ศส", f"Expected 'ศส', got {res['d3']}"
            assert res["d4"] == "1009", f"Expected '1009', got {res['d4']}"
            assert res["province"] == "กรุงเทพมหานคร", f"Expected 'กรุงเทพมหานคร', got {res['province']}"
            print(f"  -> Real image a1.3.4.1 inference OK: {res['full_plate']} (Confidence: {res['confidence']:.2%})")
    else:
        print(f"  -> Checkpoint not found at {reader.weights_path}; skipping real weight inference.")

    # 1.1 Test Khor-Kai ('ข') token mapping & inference
    from app.plate_reader import decode_tokens_to_plate, EASYOCR_DECODING_LIST
    assert len(EASYOCR_DECODING_LIST) == 187, f"Expected 187 classes, got {len(EASYOCR_DECODING_LIST)}"
    assert EASYOCR_DECODING_LIST[183] == "ข", f"Token 183 must decode to 'ข', got {EASYOCR_DECODING_LIST[183]!r}"
    assert EASYOCR_DECODING_LIST[184] == "ซ", f"Token 184 must decode to 'ซ', got {EASYOCR_DECODING_LIST[184]!r}"

    d3_test, d4_test, prov_test, _ = decode_tokens_to_plate([183, 28, 60, 3, 3, 11, 60, 93, 1])
    assert d3_test == "ขง", f"Expected 'ขง', got {d3_test!r}"
    assert d4_test == "119", f"Expected '119', got {d4_test!r}"
    assert prov_test == "กรุงเทพมหานคร", f"Expected 'กรุงเทพมหานคร', got {prov_test!r}"

    d3_khor, _, _, _ = decode_tokens_to_plate([3, 183, 183, 60, 11, 9, 11, 9, 60, 93, 1])
    assert d3_khor == "1ขข", f"Expected '1ขข', got {d3_khor!r}"

    if reader.model is not None:
        khor_plate = Path("dataset/curated_lpr_dataset/plates/AA34_car_ขง_119_กรุงเทพมหานคร.jpg")
        if khor_plate.exists():
            res_khor = reader.predict(khor_plate)
            assert res_khor["d3"] == "ขง", f"Expected 'ขง', got {res_khor['d3']!r}"
            assert res_khor["d4"] == "119", f"Expected '119', got {res_khor['d4']!r}"
            print(f"  -> Khor-Kai ('ข') real image inference OK: {res_khor['full_plate']}")
    print("  -> Khor-Kai ('ข') decode mapping verified OK")
    
    # 2. Test ImageItem comparison helper
    item = ImageItem(Path("test_WhiteBlackCar_1กข_9999_เชียงใหม่.jpg"))
    item.set_ai_prediction("1กข", "1234", "เชียงใหม่", 0.95)
    comp = item.get_ai_comparison()
    assert comp["has_ai"] is True
    assert comp["d3_match"] is True
    assert comp["d4_match"] is False  # 9999 vs 1234
    assert comp["prov_match"] is True
    assert comp["all_match"] is False
    print("  -> AI comparison logic OK")
    
    # 3. Test Selective and Bulk Replacement in GUI
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.chk_auto_ai.setChecked(False)  # manual control for test
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_img = tmp_path / "car01_WhiteBlackCar_1กก_1111_กรุงเทพมหานคร.jpg"
        Image.new("RGB", (100, 32), color=(255, 255, 255)).save(test_img)
        
        assert window.session.load_folder(str(tmp_path))
        window._show_item_at(0)
        
        # Verify initial values
        assert window.txt_d3.text() == "1กก"
        assert window.txt_d4.text() == "1111"
        assert window.combo_province.get_province_value() == "กรุงเทพมหานคร"
        
        # Set mock AI prediction
        window._latest_ai_d3 = "2ขข"
        window._latest_ai_d4 = "2222"
        window._latest_ai_prov = "เชียงใหม่"
        window._latest_ai_conf = 0.96
        window._refresh_comparison_view()
        
        # A) Test individual replacement of d3
        window._on_apply_ai_d3()
        assert window.txt_d3.text() == "2ขข"
        assert window.txt_d4.text() == "1111"  # unchanged
        assert window.combo_province.get_province_value() == "กรุงเทพมหานคร"  # unchanged
        print("  -> Single replace d3 OK")
        
        # B) Test individual replacement of d4
        window._on_apply_ai_d4()
        assert window.txt_d3.text() == "2ขข"
        assert window.txt_d4.text() == "2222"
        assert window.combo_province.get_province_value() == "กรุงเทพมหานคร"
        print("  -> Single replace d4 OK")
        
        # C) Test individual replacement of province
        window._on_apply_ai_prov()
        assert window.combo_province.get_province_value() == "เชียงใหม่"
        print("  -> Single replace province OK")
        
        # Reset values
        window.txt_d3.setText("old_d3")
        window.txt_d4.setText("old_d4")
        window.combo_province.set_province_value("ภูเก็ต")
        
        # D) Test selective replacement via checkboxes
        window.chk_sel_d3.setChecked(True)
        window.chk_sel_d4.setChecked(False)  # do not replace d4
        window.chk_sel_prov.setChecked(True)
        window._on_replace_selected_ai()
        assert window.txt_d3.text() == "2ขข"
        assert window.txt_d4.text() == "old_d4"  # remained old
        assert window.combo_province.get_province_value() == "เชียงใหม่"
        print("  -> Selective replace OK")
        
        # E) Test Replace ALL (Ctrl+R)
        window.txt_d4.setText("old_d4")
        window._on_replace_all_ai()
        assert window.txt_d3.text() == "2ขข"
        assert window.txt_d4.text() == "2222"
        assert window.combo_province.get_province_value() == "เชียงใหม่"
        print("  -> Replace ALL (Ctrl+R) OK")
        
        window._stop_workers()

    print("  -> AI Reader Comparison & Replacement All Verified OK")


if __name__ == "__main__":
    import traceback
    try:
        test_provinces_and_plate_types()
        test_subfolders_and_parsing()
        test_delete_and_custom_rename()
        test_remove_dup_images()
        test_session_persistence()
        test_ai_reader_comparison_and_replace()
        test_gui_headless()
        print("\nALL 7 TEST SUITES PASSED!")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
