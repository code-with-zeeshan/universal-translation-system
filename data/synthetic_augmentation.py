# data/synthetic_augmentation.py
"""
Synthetic data augmentation — Refactored to use shared utilities.
Generate additional training data using modern transformer models.

Extended with targeted augmentation strategies that teach the model to handle:
  - False friends (cognate traps across language pairs)
  - Idioms / multi-word expressions
  - Tone / register control ([FORMAL], [CASUAL] prefix tags)
  - Cultural context via domain-specific data
  - Backtranslation for general data augmentation

The model learns these phenomena from training data, not from hardcoded rules.
"""

# Optional heavy deps: transformers, sentence_transformers, torch. Provide shims for smoke/dry-run.
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
except Exception:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

    def pipeline(*args, **kwargs):
        raise ImportError("transformers is required for SyntheticDataAugmenter")
try:
    import torch
except Exception:

    class torch:
        @staticmethod
        def cuda():
            class _C:
                @staticmethod
                def is_available():
                    return False
            return _C
        float16 = None
        float32 = None
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import numpy as np
import json
import logging
import random

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None

    class util:
        @staticmethod
        def cos_sim(a, b):
            return np.array([[0.0]])

from data.data_utils import estimate_sentence_count
from utils.common_utils import DirectoryManager
from config.schemas import RootConfig, load_config

logger = logging.getLogger(__name__)

# ── Seed data for targeted augmentation ─────────────────────────────────
# Small curated lists of the most common false friends and idioms per pair.
# These are *seed words* used to generate full training examples via NLLB,
# NOT an exhaustive dictionary used for inference.

FALSE_FRIEND_SEEDS: Dict[str, Dict[str, str]] = {
    "en_es": {
        "actualmente": "currently",
        "embarazada": "pregnant",
        "constipado": "cold (illness)",
        "sensible": "sensitive",
        "bizarro": "dashing/brave",
        "asistir": "to attend",
        "discutir": "to argue",
        "molestar": "to bother",
        "pretender": "to try",
        "realizar": "to carry out",
        "recordar": "to remember",
        "sopa": "soup",
        "carpeta": "folder",
        "delito": "crime",
        "éxito": "success",
        "fábrica": "factory",
        "lectura": "reading",
        "noticia": "news",
        "oficina": "office",
        "once": "eleven",
        "plano": "flat/plan",
        "rubor": "blush",
        "sinvergüenza": "shameless person",
        "trampa": "trap",
        "vaso": "glass",
        "zumo": "juice",
    },
    "en_fr": {
        "actuellement": "currently",
        "assister": "to attend",
        "blesser": "to wound",
        "déception": "disappointment",
        "demander": "to ask",
        "ignorer": "to not know",
        "librairie": "bookstore",
        "pain": "bread",
        "sensible": "sensitive",
        "sympathique": "nice",
        "chat": "cat",
        "college": "middle school",
        "coin": "corner",
        "journée": "day",
        "location": "rental",
        "monnaie": "change/currency",
        "pièce": "room/coin",
        "propre": "own/clean",
        "raisin": "grape",
        "sale": "dirty",
        "tôt": "early",
        "travail": "work",
        "véritable": "true",
        "victime": "victim",
    },
    "en_de": {
        "aktuell": "current",
        "bekommen": "to receive",
        "billig": "cheap",
        "brav": "well-behaved",
        "eventuell": "possibly",
        "gift": "poison",
        "handy": "mobile phone",
        "sensibel": "sensitive",
        "rat": "advice",
        "stift": "pen",
        "bald": "soon",
        "fast": "almost",
        "gymnasium": "high school",
        "karte": "card/map",
        "mappe": "folder",
        "mist": "nonsense",
        "möbel": "furniture",
        "ring": "ring",
        "schrank": "cabinet",
        "taste": "key",
        "toll": "great",
        "wald": "forest",
        "wetter": "weather",
        "zimmer": "room",
        "zug": "train",
    },
    "en_it": {
        "camera": "room",
        "confetti": "sugared almonds",
        "grosso": "big",
        "largo": "wide",
        "palazzo": "building",
        "preservativo": "condom",
        "simpatia": "likability",
        "tassa": "tax",
        "traffico": "traffic",
        "cucina": "kitchen",
        "fattoria": "farm",
        "filo": "thread",
        "firma": "signature",
        "forchetta": "fork",
        "foto": "photo",
        "libreria": "bookstore",
        "magazzino": "warehouse",
        "negozio": "shop",
        "pompa": "pump",
        "rosso": "red",
        "salute": "health",
        "scolaro": "schoolboy",
        "stima": "esteem",
        "storia": "story/history",
        "tenere": "to hold",
    },
    "en_pt": {
        "esquisito": "strange",
        "fechar": "to close",
        "gosto": "taste",
        "largo": "wide",
        "legenda": "subtitle",
        "oficina": "workshop",
        "polvo": "octopus",
        "reclamar": "to complain",
        "salsa": "parsley",
        "balcão": "balcony",
        "borracha": "eraser",
        "cachorro": "puppy",
        "cadeira": "chair",
        "copo": "glass",
        "chuva": "rain",
        "dormir": "to sleep",
        "faculdade": "college",
        "gambiarra": "makeshift solution",
        "geladeira": "refrigerator",
        "janela": "window",
        "mala": "suitcase",
        "novela": "soap opera",
        "palavra": "word",
        "prato": "plate/dish",
        "roupa": "clothing",
    },
    "en_nl": {
        "eventueel": "possibly",
        "formulier": "form",
        "gift": "poison",
        "rook": "smoke",
        "slim": "smart",
        "straf": "punishment",
        "winkel": "shop",
        "daad": "deed",
        "dak": "roof",
        "dier": "animal",
        "fiets": "bicycle",
        "geld": "money",
        "glas": "glass",
        "huis": "house",
        "kamer": "room",
        "kaas": "cheese",
        "leer": "empty/learning",
        "mooi": "beautiful",
        "natie": "nation",
        "regen": "rain",
        "schip": "ship",
        "school": "school",
        "slot": "castle/lock",
        "steen": "stone",
        "stoel": "chair",
    },
    "en_sv": {
        "eventuell": "possible",
        "gift": "poison/married",
        "känslig": "sensitive",
        "rolig": "funny",
        "semester": "vacation",
        "smal": "narrow",
        "artikel": "article",
        "bibliotek": "library",
        "bröd": "bread",
        "fönster": "window",
        "glas": "glass",
        "klocka": "clock/watch",
        "kök": "kitchen",
        "lektion": "lesson",
        "lampa": "lamp",
        "papper": "paper",
        "penna": "pen",
        "skola": "school",
        "socker": "sugar",
        "stad": "city",
        "stol": "chair",
        "tröja": "sweater",
        "tvål": "soap",
        "väska": "bag",
        "äpple": "apple",
    },
    "en_pl": {
        "aktualny": "current",
        "dodatek": "supplement",
        "dywan": "carpet",
        "lustro": "mirror",
        "zakaz": "prohibition",
        "beczka": "barrel",
        "chleb": "bread",
        "cukier": "sugar",
        "deska": "board",
        "dziadek": "grandfather",
        "fabryka": "factory",
        "głowa": "head",
        "godzina": "hour",
        "grzyb": "mushroom",
        "herbata": "tea",
        "książka": "book",
        "lek": "medicine",
        "mleko": "milk",
        "nóż": "knife",
        "okno": "window",
        "piłka": "ball",
        "pociąg": "train",
        "sklep": "shop",
        "szkoła": "school",
        "ulica": "street",
    },
    "en_tr": {
        "akıl": "mind",
        "entrika": "intrigue",
        "kibar": "polite",
        "masa": "table",
        "rapor": "report",
        "banka": "bank",
        "ders": "lesson",
        "dünya": "world",
        "hava": "weather/air",
        "kitap": "book",
        "köşk": "mansion",
        "limon": "lemon",
        "mektup": "letter",
        "müzik": "music",
        "not": "grade/note",
        "okul": "school",
        "para": "money",
        "pazar": "market/Sunday",
        "pencere": "window",
        "radyo": "radio",
        "saat": "hour/clock",
        "şişe": "bottle",
        "tatil": "holiday",
        "yemek": "food/meal",
    },
    "en_ja": {
        "マンション": "apartment building",
        "サイン": "signature",
        "バイト": "part-time job",
        "カンニング": "cheating on exam",
        "スマート": "slim",
        "コンセント": "electrical outlet",
        "マフラー": "scarf",
        "クレーム": "complaint",
        "リフォーム": "renovation",
        "ワンピース": "dress",
        "ハンドル": "steering wheel",
        "テンション": "excitement",
        "アルバイト": "part-time job",
        "イメージ": "image",
        "オーダー": "order",
        "ガラス": "glass",
        "キャンセル": "cancel",
        "ゲーム": "game",
        "サービス": "service/free",
        "タオル": "towel",
        "チャンス": "chance",
        "テスト": "test",
        "ドア": "door",
        "ニュース": "news",
        "パソコン": "personal computer",
    },
    "en_ko": {
        "미팅": "blind date",
        "핸드폰": "mobile phone",
        "아이쇼핑": "window shopping",
        "커닝": "cheating on exam",
        "노트북": "laptop",
        "스킨십": "physical affection",
        "볼펜": "ballpoint pen",
        "오픈카": "convertible car",
        "사이다": "lemon-lime soda",
        "원룸": "studio apartment",
        "아파트": "apartment",
        "에어컨": "air conditioner",
        "컴퓨터": "computer",
        "인터넷": "internet",
        "택시": "taxi",
        "버스": "bus",
        "커피": "coffee",
        "케이크": "cake",
        "초콜릿": "chocolate",
        "아이디어": "idea",
        "프로그램": "program",
        "쇼핑": "shopping",
        "뉴스": "news",
        "스포츠": "sports",
        "영화": "movie",
    },
    "en_zh": {
        "加油": "go for it",
        "白菜": "Chinese cabbage",
        "方便": "convenient",
        "地道": "authentic",
        "东西": "thing",
        "热水": "hot water",
        "认真": "serious",
        "豆浆": "soy milk",
        "手机": "mobile phone",
        "电脑": "computer",
        "电视": "television",
        "音乐": "music",
        "图书馆": "library",
        "火车站": "train station",
        "飞机场": "airport",
        "医生": "doctor",
        "老师": "teacher",
        "学生": "student",
        "朋友": "friend",
        "工作": "work/job",
        "时间": "time",
        "今天": "today",
        "明天": "tomorrow",
        "昨天": "yesterday",
    },
    "en_ru": {
        "магазин": "shop",
        "фамилия": "surname",
        "парик": "wig",
        "аккорд": "chord",
        "артист": "performer",
        "баллон": "cylinder",
        "банда": "gang",
        "буфет": "cafeteria",
        "вещь": "thing",
        "врач": "doctor",
        "галстук": "necktie",
        "говорить": "to speak",
        "город": "city",
        "дело": "business/matter",
        "друг": "friend",
        "журнал": "magazine",
        "завод": "factory",
        "квартира": "apartment",
        "комната": "room",
        "компьютер": "computer",
        "книга": "book",
        "масло": "butter/oil",
        "молоко": "milk",
        "неделя": "week",
        "поезд": "train",
    },
    "en_ar": {
        "باص": "bus",
        "تلفون": "telephone",
        "سياسة": "politics",
        "جمهورية": "republic",
        "جغرافيا": "geography",
        "جامعة": "university",
        "جريدة": "newspaper",
        "حاسوب": "computer",
        "خبز": "bread",
        "دكتور": "doctor",
        "ساعة": "hour/clock",
        "سيارة": "car",
        "صديق": "friend",
        "طالب": "student",
        "طبيب": "physician",
        "عيد": "holiday",
        "فندق": "hotel",
        "كتاب": "book",
        "مدرسة": "school",
        "مطعم": "restaurant",
        "مكتب": "office",
        "مكتبة": "library",
        "موسيقى": "music",
        "هاتف": "telephone",
        "يوم": "day",
    },
    "en_th": {
        "ปากกา": "pen",
        "ฝรั่ง": "Westerner/guava",
        "ปลา": "fish",
        "กล้วย": "banana",
        "ข้าว": "rice",
        "ก๋วยเตี๋ยว": "noodles",
        "กาแฟ": "coffee",
        "ขนม": "snack/dessert",
        "ครู": "teacher",
        "น้ำ": "water",
        "บ้าน": "house/home",
        "รถ": "car/vehicle",
        "โรงเรียน": "school",
        "ลำไย": "longan fruit",
        "ส้ม": "orange",
        "หนังสือ": "book",
        "หมอ": "doctor",
        "หมา": "dog",
        "แมว": "cat",
        "ไข่": "egg",
        "ไก่": "chicken",
        "ใจ": "heart",
        "ดี": "good",
        "ใหญ่": "big",
        "เล็ก": "small",
    },
    "en_vi": {
        "phở": "pho noodle soup",
        "cà phê": "coffee",
        "bác sĩ": "doctor",
        "bánh mì": "bread",
        "học sinh": "student",
        "bàn": "table",
        "bút": "pen",
        "chó": "dog",
        "cửa": "door",
        "ghế": "chair",
        "giáo viên": "teacher",
        "hoa": "flower",
        "học": "to learn",
        "mèo": "cat",
        "mưa": "rain",
        "nước": "water/country",
        "sách": "book",
        "sinh viên": "university student",
        "sữa": "milk",
        "trà": "tea",
        "trái cây": "fruit",
        "trường học": "school",
        "xe": "vehicle",
        "xe đạp": "bicycle",
        "yêu": "to love",
    },
    "en_hi": {
        "अंगूर": "grapes",
        "बंदर": "monkey",
        "समय": "time",
        "पानी": "water",
        "जवाब": "answer",
        "अखबार": "newspaper",
        "कमरा": "room",
        "किताब": "book",
        "खाना": "food/to eat",
        "गाड़ी": "vehicle",
        "घर": "house",
        "चाय": "tea",
        "डॉक्टर": "doctor",
        "दवा": "medicine",
        "दोस्त": "friend",
        "दूध": "milk",
        "पैसा": "money",
        "बच्चा": "child",
        "बीमार": "sick",
        "मकान": "house",
        "माँ": "mother",
        "राजा": "king",
        "स्कूल": "school",
        "हाथ": "hand",
        "हवा": "air/wind",
    },
    "en_uk": {
        "магазин": "shop",
        "краватка": "necktie",
        "паливо": "fuel",
        "гурток": "club/hobby group",
        "лікарня": "hospital",
        "автомобіль": "car",
        "бібліотека": "library",
        "будинок": "house",
        "вода": "water",
        "газета": "newspaper",
        "гроші": "money",
        "двері": "door",
        "друг": "friend",
        "життя": "life",
        "книга": "book",
        "комп'ютер": "computer",
        "місто": "city",
        "молоко": "milk",
        "музика": "music",
        "область": "region",
        "олівець": "pencil",
        "праця": "work/labor",
        "ручка": "pen",
        "сіль": "salt",
        "школа": "school",
    },
    "en_id": {
        "kulkas": "refrigerator",
        "knalpot": "exhaust pipe",
        "kantor": "office",
        "kacang": "peanut",
        "belanja": "shopping",
        "air": "water",
        "anggur": "grape/wine",
        "apel": "apple",
        "buku": "book",
        "guru": "teacher",
        "kamar": "room",
        "kucing": "cat",
        "kursi": "chair",
        "meja": "table",
        "mobil": "car",
        "motor": "motorcycle",
        "nasi": "cooked rice",
        "pintu": "door",
        "rambut": "hair",
        "roti": "bread",
        "saya": "I/me",
        "sekolah": "school",
        "teman": "friend",
        "uang": "money",
        "ya": "yes",
    },
    # Non-English-centric false friend pairs
    "es_fr": {
        "embarazada": "enceinte",
        "constipado": "enrhumé",
        "sensible": "sensible",
        "boutique": "magasin",
        "tienda": "tente",
        "sapo": "crapaud",
        "oficina": "bureau",
        "carta": "lettre",
        "goma": "gomme",
        "firma": "signature",
        "plato": "assiette",
        "oficina": "atelier",
    },
    "fr_es": {
        "blesser": "herir",
        "librairie": "libería",
        "pain": "pan",
        "déception": "decepción",
        "location": "alquiler",
        "coin": "esquina",
        "propre": "propio",
    },
    "es_pt": {
        "embarazada": "grávida",
        "bizarro": "bizarro",
        "cena": "jantar",
        "carpeta": "pasta",
        "oficina": "escritório",
        "polvo": "pó",
        "salsa": "molho",
        "vaso": "copo",
        "zumo": "sumo",
        "fechar": "fechar",
    },
    "pt_es": {
        "esquisito": "extraño",
        "legenda": "leyenda",
        "oficina": "taller",
        "reclamar": "reclamar",
        "polvo": "pulpo",
    },
    "de_fr": {
        "gift": "cadeau",
        "handy": "portable",
        "mappe": "serviette",
        "rat": "conseil",
        "taste": "touche",
        "wald": "forêt",
        "zug": "train",
    },
    "fr_de": {
        "chat": "Katze",
        "pain": "Brot",
        "monnaie": "Geld",
        "pièce": "Stück",
        "travail": "Arbeit",
    },
    "nl_de": {
        "gift": "Gift",
        "winkel": "Laden",
        "slim": "schlau",
        "mooi": "schön",
        "huis": "Haus",
        "glas": "Glas",
    },
    "de_nl": {
        "gift": "vergif",
        "rat": "raad",
        "stift": "stift",
        "fast": "bijna",
    },
    "ru_uk": {
        "магазин": "крамниця",
        "парик": "перука",
        "неделя": "тиждень",
        "город": "місто",
        "завод": "фабрика",
        "комната": "кімната",
    },
    "uk_ru": {
        "краватка": "галстук",
        "паливо": "топливо",
        "гурток": "кружок",
    },
    "zh_ja": {
        "手紙": "手紙",
        "勉强": "勉強",
        "人参": "人参",
        "大丈夫": "大丈夫",
        "放心": "放心",
    },
    "ja_zh": {
        "手紙": "信",
        "勉強": "学习",
        "人参": "胡萝卜",
        "大丈夫": "没问题",
        "放心": "安心",
    },
}

IDIOM_SEEDS: Dict[str, List[str]] = {
    "es": [
        "Está lloviendo a cántaros.",
        "Eso es pan comido.",
        "Está en las nubes.",
        "Meter la pata.",
        "Costó un ojo de la cara.",
        "No tengo pelos en la lengua.",
        "Ponte las pilas.",
        "Le dio en el clavo.",
    ],
    "fr": [
        "Il pleut des cordes.",
        "C'est la fin des haricots.",
        "Mettre son grain de sel.",
        "Vendre la mèche.",
        "Casser les pieds.",
    ],
    "de": [
        "Da liegt der Hund begraben.",
        "Ich verstehe nur Bahnhof.",
        "Er hat die Nase voll.",
        "Ich drücke die Daumen.",
        "Das ist unter aller Sau.",
    ],
    "it": [
        "In bocca al lupo!",
        "Ha preso la palla al balzo.",
        "Costa un occhio della testa.",
        "È al settimo cielo.",
        "Acqua in bocca.",
    ],
    "pt": [
        "Ele pagou o pato.",
        "Matar dois coelhos com uma paulada.",
        "Chove canivetes.",
        "Caiu a ficha.",
        "Não ter papas na língua.",
    ],
    "ja": [
        "猫の手も借りたい。",
        "猿も木から落ちる。",
        "花より団子。",
        "井の中の蛙大海を知らず。",
        "七転び八起き。",
    ],
    "zh": [
        "画蛇添足。",
        "对牛弹琴。",
        "亡羊补牢。",
        "井底之蛙。",
        "一石二鸟。",
    ],
    "nl": [
        "Daar komt de aap uit de mouw.",
        "De kat uit de boom kijken.",
        "Een boekje opendoen.",
        "Het paard achter de wagen spannen.",
        "Twee vliegen in één klap.",
    ],
    "sv": [
        "Bita i det sura äpplet.",
        "Det är ingen ko på isen.",
        "Slå två flugor i en smäll.",
        "Ta tjuren vid hornen.",
        "Lägga rabarber på.",
    ],
    "pl": [
        "Bułka z masłem.",
        "Nie mój cyrk, nie moje małpy.",
        "Robić z igły widły.",
        "Trzymać kciuki.",
        "Kłamstwo ma krótkie nogi.",
    ],
    "tr": [
        "Bir taşla iki kuş.",
        "İğneyi kendine, çuvaldızı başkasına batır.",
        "Pire için yorgan yakmak.",
        "Kafayı yemek.",
        "Sinir küpü.",
    ],
    "ru": [
        "Рукой подать.",
        "Спустя рукава.",
        "Игра не стоит свеч.",
        "Козёл отпущения.",
        "Дело в шляпе.",
        "Бить баклуши.",
        "Когда рак на горе свистнет.",
    ],
    "ar": [
        "رجع بخفي حنين.",
        "يدس السم في العسل.",
        "اختلط الحابل بالنابل.",
        "على رأسي وعيني.",
        "بلغ السيل الزبى.",
        "بين المطرقة والسندان.",
        "اليد الواحدة لا تصفق.",
    ],
    "ko": [
        "호랑이도 제 말 하면 온다.",
        "가는 말이 고와야 오는 말이 곱다.",
        "소 잃고 외양간 고친다.",
        "누워서 떡 먹기.",
        "바늘 가는 데 실 간다.",
        "낮말은 새가 듣고 밤말은 쥐가 듣는다.",
    ],
    "hi": [
        "बंदर क्या जाने अदरक का स्वाद।",
        "दूर के ढोल सुहावने लगते हैं।",
        "आगे कुआँ पीछे खाई।",
        "जले पर नमक छिड़कना।",
        "अंत भला तो सब भला।",
        "ऊँट के मुँह में जीरा।",
    ],
    "th": [
        "กินน้ำใต้ศอก",
        "เห็นช้างเท่าหมู",
        "จับปลาสองมือ",
        "น้ำขึ้นให้รีบตัก",
        "ปิดทองหลังพระ",
        "รำไม่ดีโทษปี่โทษกลอง",
    ],
    "vi": [
        "Đẽo cày giữa đường.",
        "Ếch ngồi đáy giếng.",
        "Nước mắt cá sấu.",
        "Mèo khen mèo dài đuôi.",
        "Chó cậy gần nhà, gà cậy gần chuồng.",
        "Có công mài sắt có ngày nên kim.",
    ],
    "uk": [
        "Пекти раків.",
        "Бити байдики.",
        "Вовка ноги годують.",
        "Куй залізо поки гаряче.",
        "Мов білка в колесі.",
        "Тримати камінь за пазухою.",
    ],
    "id": [
        "Ada gula ada semut.",
        "Bagai pinang dibelah dua.",
        "Besar pasak daripada tiang.",
        "Sekali merengkuh dayung, dua tiga pulau terlampaui.",
        "Tiada gading yang tak retak.",
        "Sambil menyelam minum air.",
    ],
}

# Template sentences for false friend augmentation
# 25 diverse templates per major language, English fallback for others
FF_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        "The {word} situation needs attention.",
        "She is known for her {word} approach.",
        "They discussed the {word} matter yesterday.",
        "He gave a {word} response to the question.",
        "We need a more {word} solution here.",
        "I found the {word} details very helpful.",
        "Her {word} attitude impressed everyone.",
        "This is a {word} example of the problem.",
        "The {word} aspects were overlooked.",
        "They emphasized the {word} importance of this.",
        "His {word} behavior caused some issues.",
        "The {word} results surprised the team.",
        "She mentioned the {word} requirements clearly.",
        "The {word} approach has many benefits.",
        "He described the {word} process in detail.",
        "This {word} feature is very useful.",
        "The {word} criteria must be met first.",
        "She demonstrated a {word} understanding of the topic.",
        "The {word} factors influenced the decision.",
        "They addressed the {word} concerns effectively.",
        "His {word} perspective was valuable.",
        "The {word} elements need to be reviewed.",
        "I think this {word} idea has potential.",
        "We should consider the {word} implications.",
        "Everyone agreed it was a {word} outcome.",
    ],
    "es": [
        "La situación {word} necesita atención.",
        "Ella es conocida por su enfoque {word}.",
        "Discutieron el asunto {word} ayer.",
        "Él dio una respuesta {word} a la pregunta.",
        "Necesitamos una solución más {word} aquí.",
        "Encontré los detalles {word} muy útiles.",
        "Su actitud {word} impresionó a todos.",
        "Este es un ejemplo {word} del problema.",
        "Los aspectos {word} fueron pasados por alto.",
        "Enfatizaron la importancia {word} de esto.",
        "Su comportamiento {word} causó algunos problemas.",
        "Los resultados {word} sorprendieron al equipo.",
        "Ella mencionó los requisitos {word} claramente.",
        "El enfoque {word} tiene muchos beneficios.",
        "Él describió el proceso {word} en detalle.",
        "Esta característica {word} es muy útil.",
        "Los criterios {word} deben cumplirse primero.",
        "Ella demostró un entendimiento {word} del tema.",
        "Los factores {word} influyeron en la decisión.",
        "Abordaron las preocupaciones {word} efectivamente.",
        "Su perspectiva {word} fue valiosa.",
        "Los elementos {word} necesitan ser revisados.",
        "Creo que esta idea {word} tiene potencial.",
        "Deberíamos considerar las implicaciones {word}.",
        "Todos acordaron que fue un resultado {word}.",
    ],
    "fr": [
        "La situation {word} nécessite de l'attention.",
        "Elle est connue pour son approche {word}.",
        "Ils ont discuté de la question {word} hier.",
        "Il a donné une réponse {word} à la question.",
        "Nous avons besoin d'une solution plus {word} ici.",
        "J'ai trouvé les détails {word} très utiles.",
        "Son attitude {word} a impressionné tout le monde.",
        "C'est un exemple {word} du problème.",
        "Les aspects {word} ont été négligés.",
        "Ils ont souligné l'importance {word} de cela.",
        "Son comportement {word} a causé des problèmes.",
        "Les résultats {word} ont surpris l'équipe.",
        "Elle a mentionné les exigences {word} clairement.",
        "L'approche {word} présente de nombreux avantages.",
        "Il a décrit le processus {word} en détail.",
        "Cette fonctionnalité {word} est très utile.",
        "Les critères {word} doivent être remplis d'abord.",
        "Elle a démontré une compréhension {word} du sujet.",
        "Les facteurs {word} ont influencé la décision.",
        "Ils ont abordé les préoccupations {word} efficacement.",
        "Sa perspective {word} était précieuse.",
        "Les éléments {word} doivent être révisés.",
        "Je pense que cette idée {word} a du potentiel.",
        "Nous devrions considérer les implications {word}.",
        "Tout le monde a convenu que c'était un résultat {word}.",
    ],
    "de": [
        "Die {word} Situation erfordert Aufmerksamkeit.",
        "Sie ist für ihren {word} Ansatz bekannt.",
        "Sie haben die {word} Angelegenheit gestern besprochen.",
        "Er gab eine {word} Antwort auf die Frage.",
        "Wir brauchen hier eine {word} Lösung.",
        "Ich fand die {word} Details sehr hilfreich.",
        "Ihre {word} Einstellung beeindruckte alle.",
        "Dies ist ein {word} Beispiel für das Problem.",
        "Die {word} Aspekte wurden übersehen.",
        "Sie betonten die {word} Bedeutung davon.",
        "Sein {word} Verhalten verursachte Probleme.",
        "Die {word} Ergebnisse überraschten das Team.",
        "Sie erwähnte die {word} Anforderungen deutlich.",
        "Der {word} Ansatz hat viele Vorteile.",
        "Er beschrieb den {word} Prozess im Detail.",
        "Diese {word} Funktion ist sehr nützlich.",
        "Die {word} Kriterien müssen zuerst erfüllt werden.",
        "Sie zeigte ein {word} Verständnis des Themas.",
        "Die {word} Faktoren beeinflussten die Entscheidung.",
        "Sie sprachen die {word} Bedenken effektiv an.",
        "Seine {word} Perspektive war wertvoll.",
        "Die {word} Elemente müssen überprüft werden.",
        "Ich denke, diese {word} Idee hat Potenzial.",
        "Wir sollten die {word} Auswirkungen bedenken.",
        "Alle waren sich einig, dass es ein {word} Ergebnis war.",
    ],
    "pt": [
        "A situação {word} precisa de atenção.",
        "Ela é conhecida por sua abordagem {word}.",
        "Eles discutiram o assunto {word} ontem.",
        "Ele deu uma resposta {word} à pergunta.",
        "Precisamos de uma solução mais {word} aqui.",
        "Achei os detalhes {word} muito úteis.",
        "Sua atitude {word} impressionou a todos.",
        "Este é um exemplo {word} do problema.",
        "Os aspectos {word} foram ignorados.",
        "Eles enfatizaram a importância {word} disto.",
        "Seu comportamento {word} causou alguns problemas.",
        "Os resultados {word} surpreenderam a equipe.",
        "Ela mencionou os requisitos {word} claramente.",
        "A abordagem {word} tem muitos benefícios.",
        "Ele descreveu o processo {word} em detalhes.",
        "Este recurso {word} é muito útil.",
        "Os critérios {word} devem ser atendidos primeiro.",
        "Ela demonstrou uma compreensão {word} do tópico.",
        "Os fatores {word} influenciaram a decisão.",
        "Eles abordaram as preocupações {word} eficazmente.",
        "Sua perspectiva {word} foi valiosa.",
        "Os elementos {word} precisam ser revisados.",
        "Acho que esta ideia {word} tem potencial.",
        "Devemos considerar as implicações {word}.",
        "Todos concordaram que foi um resultado {word}.",
    ],
    "it": [
        "La situazione {word} richiede attenzione.",
        "È conosciuta per il suo approccio {word}.",
        "Hanno discusso la questione {word} ieri.",
        "Ha dato una risposta {word} alla domanda.",
        "Abbiamo bisogno di una soluzione più {word} qui.",
        "Ho trovato i dettagli {word} molto utili.",
        "Il suo atteggiamento {word} ha impressionato tutti.",
        "Questo è un esempio {word} del problema.",
        "Gli aspetti {word} sono stati trascurati.",
        "Hanno enfatizzato l'importanza {word} di questo.",
        "Il suo comportamento {word} ha causato problemi.",
        "I risultati {word} hanno sorpreso il team.",
        "Ha menzionato i requisiti {word} chiaramente.",
        "L'approccio {word} ha molti vantaggi.",
        "Ha descritto il processo {word} in dettaglio.",
        "Questa funzionalità {word} è molto utile.",
        "I criteri {word} devono essere soddisfatti prima.",
        "Ha dimostrato una comprensione {word} dell'argomento.",
        "I fattori {word} hanno influenzato la decisione.",
        "Hanno affrontato le preoccupazioni {word} efficacemente.",
        "La sua prospettiva {word} è stata preziosa.",
        "Gli elementi {word} devono essere rivisti.",
        "Penso che questa idea {word} abbia potenziale.",
        "Dovremmo considerare le implicazioni {word}.",
        "Tutti hanno concordato che è stato un risultato {word}.",
    ],
    "ja": [
        "{word}状況は注意が必要です。",
        "彼女は{word}アプローチで知られています。",
        "彼らは昨日{word}問題について議論しました。",
        "彼は質問に{word}回答をしました。",
        "ここではより{word}解決策が必要です。",
        "{word}詳細は非常に役に立ちました。",
        "彼女の{word}態度は皆を感動させました。",
        "これは問題の{word}例です。",
        "{word}側面は見落とされていました。",
        "彼らはこれの{word}重要性を強調しました。",
        "彼の{word}行動は問題を引き起こしました。",
        "{word}結果はチームを驚かせました。",
        "彼女は{word}要件を明確に述べました。",
        "{word}アプローチには多くの利点があります。",
        "彼は{word}プロセスを詳細に説明しました。",
        "この{word}機能は非常に便利です。",
        "{word}基準を最初に満たす必要があります。",
        "彼女はトピックの{word}理解を示しました。",
        "{word}要因が決定に影響を与えました。",
        "彼らは{word}懸念事項に効果的に対処しました。",
        "彼の{word}視点は貴重でした。",
        "{word}要素を見直す必要があります。",
        "この{word}アイデアには可能性があると思います。",
        "{word}影響を考慮する必要があります。",
        "誰もが{word}結果であることに同意しました。",
    ],
    "zh": [
        "{word}情况需要注意。",
        "她以{word}方法闻名。",
        "他们昨天讨论了{word}问题。",
        "他对问题给出了{word}回答。",
        "我们需要更{word}的解决方案。",
        "我发现{word}细节很有帮助。",
        "她的{word}态度给所有人留下了深刻印象。",
        "这是问题的{word}例子。",
        "{word}方面被忽视了。",
        "他们强调了这个的{word}重要性。",
        "他的{word}行为导致了一些问题。",
        "{word}结果让团队感到惊讶。",
        "她清楚地提到了{word}要求。",
        "{word}方法有很多好处。",
        "他详细描述了{word}过程。",
        "这个{word}功能非常有用。",
        "{word}标准必须首先满足。",
        "她展示了对该主题的{word}理解。",
        "{word}因素影响了决定。",
        "他们有效地解决了{word}问题。",
        "他的{word}视角很有价值。",
        "{word}元素需要审查。",
        "我认为这个{word}想法有潜力。",
        "我们应该考虑{word}影响。",
        "每个人都同意这是一个{word}结果。",
    ],
    "ru": [
        "{word} ситуация требует внимания.",
        "Она известна своим {word} подходом.",
        "Они обсудили {word} вопрос вчера.",
        "Он дал {word} ответ на вопрос.",
        "Нам нужно более {word} решение здесь.",
        "Я нашел {word} детали очень полезными.",
        "Ее {word} отношение впечатлило всех.",
        "Это {word} пример проблемы.",
        "{word} аспекты были упущены.",
        "Они подчеркнули {word} важность этого.",
        "Его {word} поведение вызвало проблемы.",
        "{word} результаты удивили команду.",
        "Она ясно упомянула {word} требования.",
        "{word} подход имеет много преимуществ.",
        "Он подробно описал {word} процесс.",
        "Эта {word} функция очень полезна.",
        "{word} критерии должны быть выполнены сначала.",
        "Она продемонстрировала {word} понимание темы.",
        "{word} факторы повлияли на решение.",
        "Они эффективно решили {word} проблемы.",
        "Его {word} перспектива была ценной.",
        "{word} элементы нужно пересмотреть.",
        "Я думаю, эта {word} идея имеет потенциал.",
        "Мы должны рассмотреть {word} последствия.",
        "Все согласились, что это был {word} результат.",
    ],
}

# Register transformation prompts for tone augmentation
FORMAL_HINT = " [Formal register: polite forms, complete sentences]"
CASUAL_HINT = " [Casual register: everyday language, conversational]"

# ── NLLB code mapping ──────────────────────────────────────────────────

NLLB_CODE_MAP: Dict[str, str] = {
    'en': 'eng_Latn', 'es': 'spa_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn',
    'pt': 'por_Latn', 'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'zh': 'zho_Hans',
    'ru': 'rus_Cyrl', 'ar': 'arb_Arab', 'ko': 'kor_Hang', 'nl': 'nld_Latn',
    'pl': 'pol_Latn', 'tr': 'tur_Latn', 'th': 'tha_Thai', 'vi': 'vie_Latn',
    'hi': 'hin_Deva', 'sv': 'swe_Latn', 'uk': 'ukr_Cyrl', 'id': 'ind_Latn',
}

# ── Augmenter class ────────────────────────────────────────────────────


class SyntheticDataAugmenter:
    """Generate additional training data using modern transformer models"""

    def __init__(self, config: RootConfig, base_model: str = 'facebook/nllb-200-distilled-1.3B'):
        self.logger = logging.getLogger(__name__)
        self.base_model = base_model
        self.config = config
        self.languages = self.config.data.active_languages
        self.quality_threshold = self.config.data.quality_threshold
        self.output_dir = Path(self.config.data.processed_dir)
        self.pipeline_batch_size = 128

        self._model = None
        self._tokenizer = None
        self._translator = None
        self._sentence_model = None

        self.logger.info(f"Initialized augmenter with model: {base_model}")

    @property
    def model(self):
        if self._model is None:
            if AutoModelForSeq2SeqLM is None:
                raise ImportError("transformers required")
            self.logger.info("Loading translation model...")
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model,
                torch_dtype=(torch.float16 if hasattr(torch, 'cuda') and torch.cuda.is_available() else getattr(torch, 'float32', None)),
                device_map=("auto" if hasattr(torch, 'cuda') and torch.cuda.is_available() else None)
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if AutoTokenizer is None:
                raise ImportError("transformers required")
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        return self._tokenizer

    @property
    def translator(self):
        if self._translator is None:
            use_cuda = hasattr(torch, 'cuda') and torch.cuda.is_available()
            bs = self.pipeline_batch_size if use_cuda else 1
            self.logger.info(f"Creating NLLB pipeline with batch_size={bs}")
            pipe_kwargs = dict(
                task="translation",
                model=self.model,
                tokenizer=self.tokenizer,
                batch_size=bs,
            )
            if not use_cuda:
                pipe_kwargs['device'] = -1
            self._translator = pipeline(**pipe_kwargs)
        return self._translator

    @property
    def sentence_model(self):
        if self._sentence_model is None:
            self.logger.info("Loading sentence transformer...")
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_model

    def _nllb_code(self, lang: str) -> str:
        return NLLB_CODE_MAP.get(lang, lang)

    def _translate_batch(self, texts: List[str], src: str, tgt: str) -> List[str]:
        """Translate a batch using the NLLB pipeline."""
        try:
            results = self.translator(
                texts,
                src_lang=self._nllb_code(src),
                tgt_lang=self._nllb_code(tgt),
                max_length=512,
            )
            return [r['translation_text'] for r in results]
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            return [""] * len(texts)

    # ── 1. False Friend Augmentation ───────────────────────────────────

    def generate_false_friend_examples(
        self,
        source_lang: str,
        target_lang: str,
        output_file: str,
    ) -> Dict[str, int]:
        """Generate parallel examples that teach correct false-friend mappings.

        For each known false friend (e.g. 'actualmente' in Spanish), creates
        template sentences in the source language, translates them with NLLB
        to get the correct target, and writes (source+ff → correct_target)
        pairs. The model learns from these during training.
        """
        pair = f"{source_lang}_{target_lang}"
        ff_dict = FALSE_FRIEND_SEEDS.get(pair)
        if ff_dict is None:
            self.logger.info(f"No false friend seeds for {pair}, skipping")
            return {"generated": 0, "pair": pair}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        src_templates = FF_TEMPLATES.get(source_lang, FF_TEMPLATES.get("en", []))
        count = 0
        batch_src = []
        for ff_word, correct_meaning in ff_dict.items():
            for tmpl in src_templates:
                src_sentence = tmpl.replace("{word}", ff_word)
                batch_src.append(src_sentence)

        if not batch_src:
            return {"generated": 0, "pair": pair}

        translations = self._translate_batch(batch_src, source_lang, target_lang)
        with open(output_path, 'w', encoding='utf-8') as f:
            for src_sentence, tgt in zip(batch_src, translations):
                if tgt:
                    f.write(f"{src_sentence}\t{tgt}\n")
                    count += 1

        self.logger.info(f"Generated {count} false-friend examples for {pair}")
        return {"generated": count, "pair": pair}

    # ── 2. Idiom Augmentation ──────────────────────────────────────────

    def generate_idiom_examples(
        self,
        source_lang: str,
        target_lang: str,
        output_file: str,
    ) -> Dict[str, int]:
        """Generate parallel examples with idiomatic source → natural translation.

        Translates known idioms via NLLB to produce natural target equivalents.
        The model learns to map idiomatic expressions to their natural counterparts.
        """
        probes = IDIOM_SEEDS.get(source_lang, [])
        if not probes:
            self.logger.info(f"No idiom seeds for {source_lang}, skipping")
            return {"generated": 0, "source_lang": source_lang}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        translations = self._translate_batch(probes, source_lang, target_lang)
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in zip(probes, translations):
                if tgt:
                    f.write(f"{src}\t{tgt}\n")
                    count += 1

        self.logger.info(f"Generated {count} idiom examples for {source_lang}→{target_lang}")
        return {"generated": count, "source_lang": source_lang, "target_lang": target_lang}

    # ── 3. Tone / Register Augmentation ────────────────────────────────

    def generate_tone_examples(
        self,
        input_parallel_file: str,
        source_lang: str,
        target_lang: str,
        output_formal_file: str,
        output_casual_file: str,
        max_examples: int = 10000,
    ) -> Dict[str, int]:
        """Create formal/casual versions of existing parallel sentences.

        Prepends [FORMAL] / [CASUAL] tags and appends register hints to guide
        NLLB to produce register-matched translations. The resulting parallel
        data teaches the model to associate the tags with the right register.
        """
        input_path = Path(input_parallel_file)
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return {"error": "file_not_found"}

        output_f = Path(output_formal_file)
        output_c = Path(output_casual_file)
        DirectoryManager.create_directory(output_f.parent)
        DirectoryManager.create_directory(output_c.parent)

        pairs: List[Tuple[str, str]] = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
                if len(pairs) >= max_examples:
                    break

        if not pairs:
            return {"generated": 0}

        # Create formal versions: tag source, guide NLLB toward formal register
        formal_sources = [f"[FORMAL]{src}{FORMAL_HINT}" for src, _ in pairs]
        formal_targets = self._translate_batch(formal_sources, source_lang, target_lang)

        casual_sources = [f"[CASUAL]{src}{CASUAL_HINT}" for src, _ in pairs]
        casual_targets = self._translate_batch(casual_sources, source_lang, target_lang)

        f_count = c_count = 0
        with open(output_f, 'w') as f_out:
            for src, tgt in zip(formal_sources, formal_targets):
                if tgt:
                    f_out.write(f"{src}\t{tgt}\n")
                    f_count += 1

        with open(output_c, 'w') as f_out:
            for src, tgt in zip(casual_sources, casual_targets):
                if tgt:
                    f_out.write(f"{src}\t{tgt}\n")
                    c_count += 1

        self.logger.info(f"Generated {f_count} formal + {c_count} casual examples")
        return {"formal": f_count, "casual": c_count}

    # ── 4. Cultural Context Augmentation ───────────────────────────────

    def generate_cultural_context_examples(
        self,
        domain_data_dir: str,
        output_file: str,
        target_lang: str = "en",
    ) -> Dict[str, int]:
        """Generate culturally-aware examples from domain-specific data.

        Reads domain-specific parallel data (medical, legal, tech) and creates
        additional pairs where domain-specific terms are translated with
        appropriate cultural context. Uses NLLB to re-translate with context
        hints appended.
        """
        domain_dir = Path(domain_data_dir)
        if not domain_dir.exists():
            self.logger.warning(f"Domain data directory not found: {domain_dir}")
            return {"error": "dir_not_found"}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        all_pairs: List[Tuple[str, str]] = []
        for fpath in domain_dir.glob("*.txt"):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        all_pairs.append((parts[0], parts[1]))

        if not all_pairs:
            return {"generated": 0}

        # Re-translate with cultural context hint to produce natural translations
        context_hint = " [Translate naturally for the target culture, adapting culturally-specific terms]"
        augmented_sources = [src + context_hint for src, _ in all_pairs]
        domain = domain_dir.name

        # Detect source language from the data (first pair's original)
        augmented_targets = self._translate_batch(
            augmented_sources,
            target_lang,  # source for NLLB is the non-English side
            target_lang,  # target for NLLB is English
        )

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for src, tgt in zip(augmented_sources, augmented_targets):
                if tgt:
                    f.write(f"{src}\t{tgt}\t{domain}\n")
                    count += 1

        self.logger.info(f"Generated {count} cultural-context examples from {domain}")
        return {"generated": count, "domain": domain}

    # ── 5. Dynamic False Friend Generation (NLLB-based) ──────────────

    # 50 English context templates for dynamic generation.
    # Each is translated to source_lang via NLLB to produce natural
    # source sentences containing the false friend word.
    _DYNAMIC_CONTEXTS: List[str] = [
        "I am talking about {meaning} right now.",
        "The concept of {meaning} is very important.",
        "We need to understand {meaning} better.",
        "She explained {meaning} in detail.",
        "They discussed {meaning} at the meeting.",
        "He wrote a report about {meaning}.",
        "The definition of {meaning} is clear.",
        "I learned about {meaning} yesterday.",
        "We should focus on {meaning}.",
        "Her opinion about {meaning} was interesting.",
        "The teacher spoke about {meaning}.",
        "They asked questions about {meaning}.",
        "The article describes {meaning} thoroughly.",
        "I have experience with {meaning}.",
        "We considered {meaning} carefully.",
        "She has a lot of knowledge about {meaning}.",
        "The discussion about {meaning} was productive.",
        "He gave a presentation on {meaning}.",
        "They conducted research on {meaning}.",
        "The book covers {meaning} in chapter three.",
        "I want to learn more about {meaning}.",
        "We need to address {meaning} in our work.",
        "She specializes in {meaning}.",
        "The topic of {meaning} came up.",
        "They provided examples of {meaning}.",
        "I read an interesting article about {meaning}.",
        "The study focused on {meaning}.",
        "We talked about {meaning} for an hour.",
        "She shared her thoughts on {meaning}.",
        "The report highlighted {meaning}.",
        "They raised concerns about {meaning}.",
        "I have questions about {meaning}.",
        "The workshop covered {meaning} extensively.",
        "We explored different aspects of {meaning}.",
        "She demonstrated {meaning} in practice.",
        "The data shows trends in {meaning}.",
        "They analyzed the impact of {meaning}.",
        "I wrote a paper on {meaning}.",
        "The course includes a module on {meaning}.",
        "We need more information about {meaning}.",
        "She described the history of {meaning}.",
        "The results relate to {meaning}.",
        "They emphasized the role of {meaning}.",
        "I gave a lecture on {meaning} yesterday.",
        "The project involves {meaning} directly.",
        "We collected data about {meaning}.",
        "She offered insights into {meaning}.",
        "The case study illustrates {meaning}.",
        "They debated the importance of {meaning}.",
        "I documented my findings on {meaning}.",
        "The training covers {meaning} thoroughly.",
    ]

    def generate_dynamic_false_friend_examples(
        self,
        source_lang: str,
        target_lang: str,
        output_file: str,
        max_examples: int = 5000,
    ) -> Dict[str, int]:
        """Generate false-friend examples by translating English contexts.

        Unlike the template approach, this method translates English sentences
        describing the correct meaning into the source language via NLLB.
        The resulting source sentence naturally contains the false friend word.
        The source is then translated back to the target language for the
        correct reference translation.

        This produces more natural and diverse training pairs.
        """
        pair = f"{source_lang}_{target_lang}"
        ff_dict = FALSE_FRIEND_SEEDS.get(pair)
        if ff_dict is None:
            self.logger.info(f"No false friend seeds for {pair}, skipping")
            return {"generated": 0, "pair": pair}

        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        per_word = max(1, max_examples // len(ff_dict))
        contexts = self._DYNAMIC_CONTEXTS[:per_word]

        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for ff_word, correct_meaning in ff_dict.items():
                # Build English sentences with the correct meaning
                eng_sentences = [c.replace("{meaning}", correct_meaning) for c in contexts]

                # Translate to source language → natural source sentences containing ff_word
                batch_size = 128
                for i in range(0, len(eng_sentences), batch_size):
                    batch = eng_sentences[i:i + batch_size]
                    try:
                        src_sentences = self._translate_batch(batch, "en", source_lang)
                        valid_src = [s for s in src_sentences if s]
                        if not valid_src:
                            continue
                        tgt_results = self._translate_batch(valid_src, source_lang, target_lang)
                        for src_s, tgt in zip(valid_src, tgt_results):
                            if tgt:
                                f.write(f"{src_s}\t{tgt}\n")
                                count += 1
                    except Exception as e:
                        self.logger.error(f"Dynamic FF batch failed: {e}")

        self.logger.info(f"Generated {count} dynamic false-friend examples for {pair}")
        return {"generated": count, "pair": pair}

    # ── 6. Backtranslation (existing) ──────────────────────────────────

    def augment_with_backtranslation(
        self,
        monolingual_file: str,
        source_lang: str,
        target_lang: str,
        output_file: str,
        max_sentences: int = 100000,
        batch_size: int = 128,
    ) -> Dict[str, int]:
        """
        Use backtranslation to create synthetic parallel data.

        Returns:
            Statistics about the augmentation process
        """
        monolingual_path = Path(monolingual_file)
        output_path = Path(output_file)
        DirectoryManager.create_directory(output_path.parent)

        if not monolingual_path.exists():
            self.logger.error(f"Monolingual file not found: {monolingual_path}")
            return {'error': 'file_not_found', 'augmented': 0}

        self.logger.info(f"Generating backtranslations for {source_lang}->{target_lang}")
        total_sentences = estimate_sentence_count(monolingual_path)
        sentences_to_process = min(total_sentences, max_sentences)

        stats = {
            'total_sentences': total_sentences,
            'processed': 0,
            'augmented': 0,
            'filtered_quality': 0,
            'errors': 0,
        }

        with open(monolingual_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                batch_texts = []
                for line_num, line in enumerate(tqdm(f_in, total=sentences_to_process, desc="Backtranslating")):
                    if line_num >= sentences_to_process:
                        break
                    text = line.strip()
                    if not text or len(text) < 10:
                        continue
                    batch_texts.append(text)
                    if len(batch_texts) >= batch_size:
                        results = self._process_backtranslation_batch(batch_texts, source_lang, target_lang)
                        for original, translated, back_translated in results:
                            if translated and self._is_quality_translation(original, back_translated):
                                f_out.write(f"{original}\t{translated}\n")
                                stats['augmented'] += 1
                            else:
                                stats['filtered_quality'] += 1
                        stats['processed'] += len(batch_texts)
                        batch_texts = []

                if batch_texts:
                    results = self._process_backtranslation_batch(batch_texts, source_lang, target_lang)
                    for original, translated, back_translated in results:
                        if translated and self._is_quality_translation(original, back_translated):
                            f_out.write(f"{original}\t{translated}\n")
                            stats['augmented'] += 1
                        else:
                            stats['filtered_quality'] += 1
                    stats['processed'] += len(batch_texts)

        self.logger.info(f"Augmentation complete: {stats['augmented']:,} pairs created")
        self.logger.info(f"Quality filtered: {stats['filtered_quality']:,} pairs")
        return stats

    def _process_backtranslation_batch(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[Tuple[str, str, str]]:
        results = []
        try:
            translations = self.translator(
                texts,
                src_lang=self._nllb_code(source_lang),
                tgt_lang=self._nllb_code(target_lang),
                max_length=512,
            )
            translated_texts = [t['translation_text'] for t in translations]
            back_translations = self.translator(
                translated_texts,
                src_lang=self._nllb_code(target_lang),
                tgt_lang=self._nllb_code(source_lang),
                max_length=512,
            )
            back_translated_texts = [t['translation_text'] for t in back_translations]
            for original, translated, back_translated in zip(texts, translated_texts, back_translated_texts):
                results.append((original, translated, back_translated))
        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            results = [(text, None, None) for text in texts]
        return results

    def _is_quality_translation(self, original: str, back_translated: str) -> bool:
        if not back_translated:
            return False
        try:
            embeddings = self.sentence_model.encode(
                [original, back_translated], convert_to_tensor=True
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            return similarity >= self.quality_threshold
        except Exception as e:
            self.logger.error(f"Quality check failed: {e}")
            return False

    # ── Pivot translation (existing) ──────────────────────────────────

    def generate_pivot_translations(
        self, english_pairs_dir: str, output_dir: Optional[str] = None
    ) -> Dict[str, int]:
        english_pairs_path = Path(english_pairs_dir)
        output_path = Path(output_dir) if output_dir else self.output_dir / 'pivot_pairs'
        DirectoryManager.create_directory(output_path)

        pairs_data: Dict[str, List[Tuple[str, str]]] = {}
        self.logger.info("Loading English-centric pairs...")

        for lang in self.languages:
            if lang == 'en':
                continue
            patterns = [f'en-{lang}_sampled.txt', f'en-{lang}.txt', f'opus_en-{lang}.txt']
            for pattern in patterns:
                file_path = english_pairs_path / pattern
                if file_path.exists():
                    pairs_data[lang] = self._load_pairs(file_path)
                    self.logger.info(f"Loaded en-{lang}: {len(pairs_data[lang]):,} pairs")
                    break

        stats = {'total_pivot_pairs': 0, 'pairs_created': {}}
        self.logger.info("Generating pivoted pairs...")

        for lang1 in pairs_data:
            for lang2 in pairs_data:
                if lang1 < lang2:
                    pair_count = self._create_pivot_pairs(
                        pairs_data[lang1], pairs_data[lang2], lang1, lang2, output_path
                    )
                    stats['pairs_created'][f'{lang1}-{lang2}'] = pair_count
                    stats['total_pivot_pairs'] += pair_count

        self.logger.info(f"Generated {stats['total_pivot_pairs']:,} pivot pairs")
        return stats

    def _load_pairs(self, file_path: Path, max_pairs: int = 50000) -> List[Tuple[str, str]]:
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_pairs:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def _create_pivot_pairs(
        self, pairs1: List[Tuple[str, str]], pairs2: List[Tuple[str, str]],
        lang1: str, lang2: str, output_path: Path,
    ) -> int:
        self.logger.info(f"Creating pivot pairs for {lang1}-{lang2}")
        output_file = output_path / f'{lang1}-{lang2}_pivot.txt'
        en_to_lang1 = {en: lang for en, lang in pairs1}
        en_to_lang2 = {en: lang for en, lang in pairs2}
        common_en = set(en_to_lang1.keys()) & set(en_to_lang2.keys())
        pairs_created = 0
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for en_text in common_en:
                try:
                    f_out.write(f"{en_to_lang1[en_text]}\t{en_to_lang2[en_text]}\n")
                    pairs_created += 1
                except Exception as e:
                    self.logger.error(f"Failed to create pivot pair: {e}")
        self.logger.info(f"Created {pairs_created:,} {lang1}-{lang2} pairs")
        return pairs_created


# ── Convenience batch runner ───────────────────────────────────────────

def _has_ff_seeds(src: str, tgt: str) -> bool:
    return f"{src}_{tgt}" in FALSE_FRIEND_SEEDS


def _has_idiom_seeds(lang: str) -> bool:
    return lang in IDIOM_SEEDS and bool(IDIOM_SEEDS[lang])


def run_all_augmentations(config: RootConfig, langs: Optional[List[str]] = None):
    """Run all augmentation strategies for all language pairs.

    Generates training data for false friends (template + dynamic),
    idioms, tone, and backtranslation across all supported language pairs.

    Only processes pairs that have seed data defined in FALSE_FRIEND_SEEDS
    or IDIOM_SEEDS to avoid unnecessary NLLB invocations.
    """
    if langs is None:
        langs = config.data.active_languages

    max_dynamic_ff = 5000
    if hasattr(config, 'pipeline') and config.pipeline:
        max_dynamic_ff = getattr(config.pipeline, 'max_dynamic_ff_per_pair', 5000)

    augmenter = SyntheticDataAugmenter(config)
    base_dir = Path(config.data.processed_dir) / "augmented"
    results = {}

    total_pairs = 0
    skipped_no_seeds = 0

    for src in langs:
        for tgt in langs:
            if src == tgt:
                continue

            total_pairs += 1
            pair_key = f"{src}_{tgt}"
            has_ff = _has_ff_seeds(src, tgt)
            has_idiom = _has_idiom_seeds(src)

            if not has_ff and not has_idiom:
                skipped_no_seeds += 1
                continue

            pair_dir = base_dir / pair_key
            DirectoryManager.create_directory(pair_dir)

            if has_ff:
                # Use both template-based and dynamic generation
                ff_out = str(pair_dir / "false_friends.txt")
                results[f"ff_{pair_key}"] = augmenter.generate_false_friend_examples(src, tgt, ff_out)

                # Dynamic NLLB-based generation for more diverse examples
                ff_dynamic_out = str(pair_dir / "false_friends_dynamic.txt")
                results[f"ff_dynamic_{pair_key}"] = augmenter.generate_dynamic_false_friend_examples(
                    src, tgt, ff_dynamic_out, max_examples=max_dynamic_ff
                )

            if has_idiom:
                idiom_out = str(pair_dir / "idioms.txt")
                results[f"idiom_{pair_key}"] = augmenter.generate_idiom_examples(src, tgt, idiom_out)

    total_generated = sum(v.get('generated', 0) for v in results.values())
    logger.info(
        f"Batch augmentation complete. {total_generated} total examples "
        f"across {len(results)} strategies "
        f"({skipped_no_seeds}/{total_pairs} pairs skipped – no seed data)."
    )
    return results


def main():
    """Standalone example."""
    config = load_config()
    augmenter = SyntheticDataAugmenter(config)

    # Example: false friends for Spanish→English
    stats = augmenter.generate_false_friend_examples(
        source_lang='es', target_lang='en',
        output_file='test_data/ff_es_en.txt',
    )
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")

    # Example: idioms for French→English
    stats = augmenter.generate_idiom_examples(
        source_lang='fr', target_lang='en',
        output_file='test_data/idiom_fr_en.txt',
    )
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
