"""
List of 77 provinces of Thailand and special annotation codes.
"""

# Special codes as requested:
# + for not license plate
# - for cannot read
SPECIAL_OPTIONS = [
    ("+", "+ (Not License Plate / ไม่ใช่ป้ายทะเบียน)"),
    ("-", "- (Cannot Read / อ่านไม่ออก)"),
]

# 77 Thailand Provinces with English romanization for bilingual search
PROVINCES_DATA = [
    ("กรุงเทพมหานคร", "Bangkok"),
    ("กระบี่", "Krabi"),
    ("กาญจนบุรี", "Kanchanaburi"),
    ("กาฬสินธุ์", "Kalasin"),
    ("กำแพงเพชร", "Kamphaeng Phet"),
    ("ขอนแก่น", "Khon Kaen"),
    ("จันทบุรี", "Chanthaburi"),
    ("ฉะเชิงเทรา", "Chachoengsao"),
    ("ชลบุรี", "Chonburi"),
    ("ชัยนาท", "Chai Nat"),
    ("ชัยภูมิ", "Chaiyaphum"),
    ("ชุมพร", "Chumphon"),
    ("เชียงราย", "Chiang Rai"),
    ("เชียงใหม่", "Chiang Mai"),
    ("ตรัง", "Trang"),
    ("ตราด", "Trat"),
    ("ตาก", "Tak"),
    ("นครนายก", "Nakhon Nayok"),
    ("นครปฐม", "Nakhon Pathom"),
    ("นครพนม", "Nakhon Phanom"),
    ("นครราชสีมา", "Nakhon Ratchasima"),
    ("นครศรีธรรมราช", "Nakhon Si Thammarat"),
    ("นครสวรรค์", "Nakhon Sawan"),
    ("นนทบุรี", "Nonthaburi"),
    ("นราธิวาส", "Narathiwat"),
    ("น่าน", "Nan"),
    ("บึงกาฬ", "Bueng Kan"),
    ("บุรีรัมย์", "Buriram"),
    ("ปทุมธานี", "Pathum Thani"),
    ("ประจวบคีรีขันธ์", "Prachuap Khiri Khan"),
    ("ปราจีนบุรี", "Prachinburi"),
    ("ปัตตานี", "Pattani"),
    ("พระนครศรีอยุธยา", "Phra Nakhon Si Ayutthaya"),
    ("พะเยา", "Phayao"),
    ("พังงา", "Phang Nga"),
    ("พัทลุง", "Phatthalung"),
    ("พิจิตร", "Phichit"),
    ("พิษณุโลก", "Phitsanulok"),
    ("เพชรบุรี", "Phetchaburi"),
    ("เพชรบูรณ์", "Phetchabun"),
    ("แพร่", "Phrae"),
    ("ภูเก็ต", "Phuket"),
    ("มหาสารคาม", "Maha Sarakham"),
    ("มุกดาหาร", "Mukdahan"),
    ("แม่ฮ่องสอน", "Mae Hong Son"),
    ("ยโสธร", "Yasothon"),
    ("ยะลา", "Yala"),
    ("ร้อยเอ็ด", "Roi Et"),
    ("ระนอง", "Ranong"),
    ("ระยอง", "Rayong"),
    ("ราชบุรี", "Ratchaburi"),
    ("ลพบุรี", "Lopburi"),
    ("ลำปาง", "Lampang"),
    ("ลำพูน", "Lamphun"),
    ("เลย", "Loei"),
    ("ศรีสะเกษ", "Sisaket"),
    ("สกลนคร", "Sakon Nakhon"),
    ("สงขลา", "Songkhla"),
    ("สตูล", "Satun"),
    ("สมุทรปราการ", "Samut Prakan"),
    ("สมุทรสงคราม", "Samut Songkhram"),
    ("สมุทรสาคร", "Samut Sakhon"),
    ("สระแก้ว", "Sa Kaeo"),
    ("สระบุรี", "Saraburi"),
    ("สิงห์บุรี", "Sing Buri"),
    ("สุโขทัย", "Sukhothai"),
    ("สุพรรณบุรี", "Suphan Buri"),
    ("สุราษฎร์ธานี", "Surat Thani"),
    ("สุรินทร์", "Surin"),
    ("หนองคาย", "Nong Khai"),
    ("หนองบัวลำภู", "Nong Bua Lamphu"),
    ("อ่างทอง", "Ang Thong"),
    ("อำนาจเจริญ", "Amnat Charoen"),
    ("อุดรธานี", "Udon Thani"),
    ("อุตรดิตถ์", "Uttaradit"),
    ("อุทัยธานี", "Uthai Thani"),
    ("อุบลราชธานี", "Ubon Ratchathani"),
]

def get_province_entries():
    """
    Returns list of tuples: (actual_value, display_text)
    Special options are placed at the very top.
    """
    entries = []
    # Special entries
    for val, disp in SPECIAL_OPTIONS:
        entries.append((val, disp))
        
    # Standard provinces
    for th, en in PROVINCES_DATA:
        entries.append((th, f"{th} ({en})"))
        
    return entries

def match_province_value(text):
    """
    Given an input string (e.g. from filename or user input),
    returns the best matching canonical value.
    """
    if not text:
        return ""
    text_clean = text.strip()
    
    if text_clean == "+" or "not" in text_clean.lower():
        return "+"
    if text_clean == "-" or "cannot" in text_clean.lower():
        return "-"
        
    for th, en in PROVINCES_DATA:
        if text_clean.lower() == th.lower() or text_clean.lower() == en.lower():
            return th
        if th in text_clean:
            return th
            
    return text_clean
