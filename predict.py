
"""
Предсказываем ориентацию текстовых кропов (0° или 180°) через RapidOCR.
Запуск: python predict.py --data test/images --sample sample_submission.csv --out submission.csv
"""

import argparse          
import random            
import sys              
from pathlib import Path
import numpy as np       
import pandas as pd      
from tqdm import tqdm    

# Проверяем версию Python ДО импорта RapidOCR.
# Если запустите на 3.13+, сразу будет понятное сообщение,
# а не невнятная ошибкой про onnxruntime
if sys.version_info >= (3, 13):        # если Python 3.13 или новее
    raise SystemExit(                   # печатаем сообщение и выходим
        "Нужен Python 3.10–3.12. На 3.13+ onnxruntime сломан.\n"
        "Создай venv с нужной версией: python3.12 -m venv .venv"
    )

# Чтобы можно было запускать скрипт из любой папки и импорты из src/ работали.
sys.path.append(str(Path(__file__).resolve().parent))

from src.io_utils import read_image                    # читаем PNG и препроцессим
from src.postprocess import clip_prob                   # обрезаем p в [0, 1]
from src.rapid_cls import OrientationClassifier         # классификатор ориентации


# Фиксированный зерно случайности, чтобы у Вас всегда был тот же результат
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Ставим seed для random и numpy."""
    random.seed(seed)          # фиксируем стандартный random
    np.random.seed(seed)       # фиксируем numpy


def parse_args() -> argparse.Namespace:
    """Разбираем то, что пользователь написал в командной строке."""
    parser = argparse.ArgumentParser(description="Определяем ориентацию текста (0°/180°).")
    # --data: путь к папке с PNG. Если не указать — возьмётся test/images.
    parser.add_argument("--data", type=Path, default=Path("test/images"),
                        help="папка с PNG-кропами")
    # --sample: путь к sample_submission.csv. Если не указать — sample_submission.csv.
    parser.add_argument("--sample", type=Path, default=Path("sample_submission.csv"),
                        help="sample_submission.csv — берём из него порядок image_id")
    # --out: куда сохранять результат. По умолчанию submission.csv.
    parser.add_argument("--out", type=Path, default=Path("submission.csv"),
                        help="куда сохранить submission.csv")
    # --no-tta: флаг для отладки. Если указан — модель работает без TTA.
    parser.add_argument("--no-tta", action="store_true",
                        help="отключить TTA (для отладки)")
    return parser.parse_args()          # возвращаем объект с полями args.data, args.out и т.д.


def predict_all(data_dir: Path, use_tta: bool = True) -> pd.DataFrame:
    """
    Гоняем классификатор по всем PNG из папки.
    Возвращаем таблицу с image_id и p_180.
    """
    clf = OrientationClassifier()       # инициализируем RapidOCR 

    paths = sorted(data_dir.glob("*.png"))     # берём все PNG и сортируем для воспроизводимости
    if not paths:                              # если папка пустая — падаем с понятной ошибкой
        raise RuntimeError(f"В {data_dir} не нашлось ни одного PNG")
    print(f"Найдено картинок: {len(paths)}")   # печатаем сколько нашли

    rows = []                          # сюда будем складывать строки результата
    for path in tqdm(paths, desc="Predict"):   # идём по всем PNG красиво:)
        try:
            img = read_image(path)                 # читаем и препроцессим картинку
            p180 = clf.predict(img, tta=use_tta)   # получаем вероятность "перевёрнут на 180°"
        except Exception as e:                     # если картинка битая — не роняем весь прогон
            print(f"[WARN] {path.name}: {e}", file=sys.stderr)
            p180 = 0.5                             # ставим нейтральное значение
        # path.stem — это имя без ".png"
        rows.append({"image_id": path.stem, "p_180": clip_prob(p180)})

    return pd.DataFrame(rows, columns=["image_id", "p_180"])   # собираем таблицу


def align_with_sample(df: pd.DataFrame, sample_path: Path) -> pd.DataFrame:
    """
    Приводим порядок image_id к sample_submission.csv.
    Иначе платформа ругается на несовпадение наборов и попытка тратится впустую.
    """
    if not sample_path.exists():               # если sample нет — предупреждаем и возвращаем как есть
        print(f"[WARN] {sample_path} не найден, выравнивание пропущено")
        return df

    sample = pd.read_csv(sample_path)                          # читаем образец
    expected = sample["image_id"].astype(str).tolist()         # список id в нужном порядке
    got, want = set(df["image_id"]), set(expected)             # сравниваем множества

    missing = want - got                       # чего у нас не хватает
    extra = got - want                         # что у нас лишнее
    if missing or extra:                       # если что-то не так — печатаем
        print(f"Пропущено: {len(missing)}; лишних: {len(extra)}")

    # reindex гарантирует тот же порядок и набор строк, что в sample
    df = df.set_index("image_id").reindex(expected).reset_index()
    df.columns = ["image_id", "p_180"]         # возвращаем имена колонок
    df["p_180"] = df["p_180"].fillna(0.5)      # на всякий случай заполняем пустые
    return df


def main() -> None:
    args = parse_args()                        # разбираем аргументы командной строки
    set_seed()                                 # фиксируем seed

    if not args.data.exists():                 # если папки с данными нет — ошибка
        raise FileNotFoundError(f"Нет такой папки: {args.data}")

    # Основной проход: препроцессинг + предсказание по всем PNG
    df = predict_all(args.data, use_tta=not args.no_tta)

    # Приводим формат к нормальному виду
    df = align_with_sample(df, args.sample)

    df.to_csv(args.out, index=False)           # сохраняем csv без колонки индексов
    print(f"\nСохранил: {args.out.resolve()}") # печатаем полный путь к файлу
    print(f"Строк: {len(df)}")                 # сколько строк получилось
    # mean(p_180) — быстрая проверка: если сильно ушло от ~0.5, что-то сломалось
    print(f"mean(p_180) = {df['p_180'].mean():.4f}")
    print(df.head())                           # показываем первые 5 строк для проверки


if __name__ == "__main__":                     # выполняется только при прямом запуске скрипта
    main()                                     # запускаем main