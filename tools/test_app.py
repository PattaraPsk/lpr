"""
Unit and Integration tests for Thai LPR Annotator with plate_type structure:
{name}_{plate_type}_{d3}_{d4}_{province}.{ext}
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
    print("[1/5] Testing Provinces & Plate Types...")
    prov_entries = get_province_entries()
    assert len(prov_entries) >= 79, f"Expected at least 79 entries, got {len(prov_entries)}"
    assert prov_entries[0][0] == "+"
    assert prov_entries[1][0] == "-"
    
    pt_entries = get_plate_type_entries()
    assert len(pt_entries) == 12 # 2 special (+, -) + 10 plate types (car & motor separated)
    assert pt_entries[0][0] == "+"
    assert pt_entries[1][0] == "-"
    
    # Test plate type matching
    assert match_plate_type("white_black_car") == "white_black_car"
    assert match_plate_type("white_black_motor") == "white_black_motor"
    assert match_plate_type("white_black") == "white_black_car" # alias
    assert match_plate_type("motorcycle") == "white_black_motor" # alias
    assert match_plate_type("white_blue") == "white_blue"
    assert match_plate_type("wbl") == "white_blue"
    assert match_plate_type("yellow_black") == "yellow_black"
    assert match_plate_type("orange_black") == "orange_black"
    assert match_plate_type("+") == "+"
    assert match_plate_type("-") == "-"
    print("  -> Provinces & Plate Types OK")

def test_session_manager():
    print("[2/5] Testing SessionManager & 5-Part Filename Parsing...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test images with {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
        img1 = tmp_path / "camera_front_white_black_car_1กก_1234_กรุงเทพมหานคร.jpg"
        img2 = tmp_path / "truck01_+_+_+_+.jpg"
        img3 = tmp_path / "van02_-_-_-_-.jpg"
        img4 = tmp_path / "bike03_white_black_motor_1ขข_999_เชียงใหม่.png"
        
        dummy_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        for p in [img1, img2, img3, img4]:
            dummy_img.save(p)
            
        session = SessionManager()
        assert session.load_folder(str(tmp_path))
        assert len(session.items) == 4
        
        # Test 5-part parsing
        item1 = next(it for it in session.items if "camera_front" in it.name)
        assert item1.name == "camera_front"
        assert item1.plate_type == "white_black_car"
        assert item1.d3 == "1กก"
        assert item1.d4 == "1234"
        assert item1.province == "กรุงเทพมหานคร"
        assert item1.status == "pending"

        
        item2 = next(it for it in session.items if it.name == "truck01")
        assert item2.plate_type == "+"
        assert item2.d3 == "+"
        assert item2.d4 == "+"
        assert item2.province == "+"
        assert item2.status == "not_plate"
        
        item3 = next(it for it in session.items if it.name == "van02")
        assert item3.plate_type == "-"
        assert item3.d3 == "-"
        assert item3.d4 == "-"
        assert item3.province == "-"
        assert item3.status == "cannot_read"
        
        item4 = next(it for it in session.items if it.name == "bike03")
        assert item4.plate_type == "white_black_motor"
        assert item4.d3 == "1ขข"
        assert item4.d4 == "999"
        assert item4.province == "เชียงใหม่"
        
        print("  -> 5-Part Parsing OK")
        
        print("[3/5] Testing Actions & Renaming with Plate Types...")
        # Test Not Plate rename on item1 -> {name}_+_+_+_+.jpg
        idx1 = session.items.index(item1)
        assert session.mark_not_plate(idx1)
        assert item1.filename == "camera_front_+_+_+_+.jpg"
        assert (tmp_path / "camera_front_+_+_+_+.jpg").exists()
        assert not (tmp_path / "camera_front_white_black_car_1กก_1234_กรุงเทพมหานคร.jpg").exists()
        assert item1.status == "not_plate"
        assert item1.plate_type == "+"
        
        # Test Cannot Read on item4 -> {name}_-_-_-_-.png
        idx4 = session.items.index(item4)
        assert session.mark_cannot_read(idx4)
        assert item4.filename == "bike03_-_-_-_-.png"
        assert (tmp_path / "bike03_-_-_-_-.png").exists()

        assert item4.status == "cannot_read"
        assert item4.plate_type == "-"
        
        # Test Edit & Approve -> {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
        assert session.approve_and_save(idx1, "white_green", "3กก", "5555", "ชลบุรี")
        assert item1.filename == "camera_front_white_green_3กก_5555_ชลบุรี.jpg"
        assert (tmp_path / "camera_front_white_green_3กก_5555_ชลบุรี.jpg").exists()
        assert item1.status == "approved"
        assert item1.plate_type == "white_green"
        
        print("  -> Actions & Renaming OK")
        
        print("[4/5] Testing Session Persistence & Resume...")
        session.current_index = 2
        session.save_session()
        
        session_file = tmp_path / ".lpr_session.json"
        assert session_file.exists(), ".lpr_session.json must exist"
        
        session2 = SessionManager()
        assert session2.load_folder(str(tmp_path))
        assert session2.current_index == 2
        
        reloaded_item1 = next(it for it in session2.items if "camera_front" in it.name)
        assert reloaded_item1.status == "approved"
        assert reloaded_item1.plate_type == "white_green"
        assert reloaded_item1.d3 == "3กก"
        assert reloaded_item1.d4 == "5555"
        assert reloaded_item1.province == "ชลบุรี"
        
        stats = session2.get_stats()
        print(f"  Stats: {stats}")
        assert stats["total"] == 4
        assert stats["approved"] == 1
        assert stats["not_plate"] == 1
        assert stats["cannot_read"] == 2
        print("  -> Persistence & Resume OK")

def test_gui_headless():
    print("[5/5] Testing GUI components with 5-part structure (offscreen)...")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6.QtWidgets import QApplication
    from app.main_window import MainWindow
    
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    
    sample_dir = str((Path(__file__).parent / "sample_images").resolve())
    assert window.session.load_folder(sample_dir)

    assert len(window.session.items) >= 10
    
    window._show_item_at(0)
    assert window.txt_name.text() != ""
    assert window.combo_plate_type.get_plate_type_value() != ""
    assert window.txt_d3.text() != ""
    assert window.txt_d4.text() != ""
    assert window.combo_province.get_province_value() != ""
    
    # Test Check View dialog
    from app.check_view import CheckViewDialog
    dlg = CheckViewDialog(window.session.items, 0, window)
    assert dlg.table.columnCount() == 8
    assert dlg.table.rowCount() == len(window.session.items)
    
    print("  -> GUI Headless Instantiation & Load OK")

if __name__ == "__main__":
    test_provinces_and_plate_types()
    test_session_manager()
    test_gui_headless()
    print("\nALL TESTS PASSED!")
