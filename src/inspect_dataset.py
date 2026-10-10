import pandas as pd


DATA_PATH = "data/raw/lenta-ru-news.csv.bz2"


def main():
    print("Загрузка датасета...")

    df = pd.read_csv(DATA_PATH)

    print("\n=== 1. Размер исходного датасета ===")
    print(df.shape)

    print("\n=== 2. Колонки ===")
    print(df.columns.tolist())

    print("\n=== 3. Первые 3 записи ===")
    print(df.head(3))

    print("\n=== 4. Типы данных ===")
    print(df.dtypes)

    print("\n=== 5. Пропущенные значения ===")
    print(df.isna().sum())

    print("\n=== 6. Количество уникальных тем ===")
    print(df["topic"].nunique())

    print("\n=== 7. Распределение новостей по темам ===")
    print(
        df["topic"]
        .value_counts(dropna=False)
        .to_string()
    )


if __name__ == "__main__":
    main()