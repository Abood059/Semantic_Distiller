#!/usr/bin/env python3
"""
Semantic Distiller - Auto Setup & Run Script
يقوم بسحب الكود من GitHub، تجهيز البيئة، وتشغيل المشروع.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ====== الإعدادات ======
REPO_URL = "https://github.com/Abood059/Semantic_Distiller.git"
REPO_DIR = "Semantic_Distiller"
CONFIG_FILE = "config.yaml"
INPUT_FILE = "test_input.json"
OUTPUT_FILE = "results.json"

# ====== الألوان للطباعة ======
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_step(msg):
    print(f"\n{BLUE}▶ {msg}{RESET}")

def print_success(msg):
    print(f"{GREEN}✓ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}✗ {msg}{RESET}")

def print_info(msg):
    print(f"{YELLOW}ℹ {msg}{RESET}")

def run_command(cmd, cwd=None, check=True):
    """تشغيل أمر في السطر وأرجاع الناتج."""
    print_info(f"تنفيذ: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), 
                           capture_output=True, text=True)
    if result.returncode != 0 and check:
        print_error(f"فشل الأمر: {result.stderr}")
        sys.exit(1)
    return result

def main():
    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}   Semantic Distiller - التشغيل التلقائي{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")

    # 1. التحقق من وجود Git
    print_step("التحقق من وجود Git...")
    if shutil.which("git") is None:
        print_error("Git غير مثبت. يرجى تثبيت Git أولاً.")
        print_info("في Ubuntu/Debian: sudo apt install git")
        print_info("في Colab: !apt-get install git")
        sys.exit(1)
    print_success("Git موجود.")

    # 2. سحب الكود من GitHub
    print_step(f"سحب الكود من {REPO_URL}...")
    if Path(REPO_DIR).exists():
        print_info(f"المجلد {REPO_DIR} موجود بالفعل. جارٍ تحديثه...")
        run_command(["git", "pull"], cwd=REPO_DIR)
    else:
        run_command(["git", "clone", REPO_URL])
    print_success("تم سحب الكود بنجاح.")

    # 3. الدخول إلى المجلد
    os.chdir(REPO_DIR)
    print_info(f"الدخول إلى: {os.getcwd()}")

    # 4. التحقق من وجود ملفات المشروع الأساسية
    print_step("التحقق من ملفات المشروع...")
    required_files = ["main.py", "run.py", "requirements.txt", "config.yaml"]
    for f in required_files:
        if not Path(f).exists():
            print_error(f"الملف {f} غير موجود في المستودع!")
            sys.exit(1)
    print_success("جميع الملفات الأساسية موجودة.")

    # 5. إنشاء البيئة الافتراضية
    print_step("إنشاء البيئة الافتراضية...")
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        run_command([sys.executable, "-m", "venv", str(venv_dir)])
        print_success("تم إنشاء البيئة الافتراضية.")
    else:
        print_info("البيئة الافتراضية موجودة بالفعل.")

    # 6. تحديد مسار Python و pip داخل البيئة الافتراضية
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"

    if not python_exe.exists():
        print_error(f"لم يتم العثور على Python في البيئة الافتراضية: {python_exe}")
        sys.exit(1)

    # 7. تثبيت التبعيات
    print_step("تثبيت التبعيات (قد يستغرق عدة دقائق)...")
    run_command([str(pip_exe), "install", "--upgrade", "pip"])
    run_command([str(pip_exe), "install", "-r", "requirements.txt"])
    print_success("تم تثبيت جميع التبعيات.")

    # 8. التحقق من ملف الإدخال
    print_step("التحقق من ملف الإدخال...")
    if not Path(INPUT_FILE).exists():
        print_info(f"ملف {INPUT_FILE} غير موجود. سيتم إنشاء ملف افتراضي.")
        # إنشاء ملف إدخال بسيط
        import json
        sample_input = {
            "initial_sentence_pool": [
                "The quick brown fox jumps over the lazy dog.",
                "A journey of a thousand miles begins with a single step.",
                "To be or not to be, that is the question.",
                "All that glitters is not gold.",
                "The early bird catches the worm.",
                "Actions speak louder than words.",
                "Beauty is in the eye of the beholder.",
                "Don't count your chickens before they hatch.",
                "Every cloud has a silver lining.",
                "Fortune favors the bold.",
                "Good things come to those who wait.",
                "Haste makes waste.",
                "Ignorance is bliss.",
                "Knowledge is power.",
                "Laughter is the best medicine.",
                "Money can't buy happiness.",
                "No pain, no gain.",
                "Practice makes perfect.",
                "Quality over quantity.",
                "Rome wasn't built in a day."
            ],
            "num_layers": 2,
            "num_nodes": 3,
            "num_output_sentences": 3,
            "prompt_template_id": "creative",
            "resume_from_checkpoint": False
        }
        with open(INPUT_FILE, "w") as f:
            json.dump(sample_input, f, indent=2)
        print_success(f"تم إنشاء {INPUT_FILE} بنجاح.")
    else:
        print_success(f"ملف {INPUT_FILE} موجود.")

    # 9. تشغيل المشروع
    print_step(f"تشغيل المشروع...")
    print_info(f"الإدخال: {INPUT_FILE}")
    print_info(f"الإخراج: {OUTPUT_FILE}")
    print_info("قد يستغرق التشغيل عدة دقائق (تحميل النماذج)...")

    cmd = [
        str(python_exe), "main.py",
        "--config", CONFIG_FILE,
        "--input", INPUT_FILE,
        "--output", OUTPUT_FILE
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print_success("تم تشغيل المشروع بنجاح!")
    except subprocess.CalledProcessError as e:
        print_error(f"فشل تشغيل المشروع: {e}")
        sys.exit(1)

    # 10. عرض النتيجة
    print_step("عرض النتيجة...")
    if Path(OUTPUT_FILE).exists():
        import json
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)
        layers = data.get("layers", [])
        print_info(f"عدد الطبقات: {len(layers)}")
        for layer in layers:
            print_info(f"  الطبقة {layer.get('layer_id')}: {len(layer.get('nodes', []))} عقد")
        print_success(f"النتائج محفوظة في: {OUTPUT_FILE}")
    else:
        print_error(f"ملف النتائج {OUTPUT_FILE} لم يتم إنشاؤه.")

    print(f"\n{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}✅ اكتمل التشغيل بنجاح!{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")

if __name__ == "__main__":
    main()