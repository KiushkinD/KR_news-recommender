"""Интерфейс для запросов и персональной ленты новостей."""
import json
from pathlib import Path
from threading import RLock

import streamlit as st

from recommender import NewsRecommender

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.8


def parse_history(text):
    return [line.strip() for line in text.splitlines() if line.strip()][-5:]


@st.cache_resource(show_spinner=False)
def load_engine(project_root):
    # Модель общая для сеансов, поэтому загрузку и вызовы защищаем блокировкой.
    engine = NewsRecommender(Path(project_root))
    positions = {aid: i for i, aid in enumerate(engine.article_ids)}
    return engine, positions, RLock()


def set_history(history):
    st.session_state.history = list(history)[-5:]
    st.session_state.history_editor = '\n'.join(st.session_state.history)
    st.session_state.last_result = None


def main():
    st.set_page_config(
        page_title='Новости по вашим интересам', page_icon='📰', layout='wide'
    )
    st.title('Новости по вашим интересам')
    st.caption('Рекомендации из архива Lenta.ru за 2018 год. Даты публикаций указаны у каждой статьи.')
    if 'history' not in st.session_state:
        st.session_state.history = []
        st.session_state.history_editor = ''
        st.session_state.last_result = None
    if 'pending_history' in st.session_state:
        st.session_state.history_editor = st.session_state.pop('pending_history')

    with st.sidebar:
        st.header('История запросов')
        st.caption('Учитываются последние пять запросов. Более ранние — сверху, новые — снизу.')
        with st.form('history_form'):
            history_text = st.text_area(
                'Один запрос на строку', key='history_editor', height=170
            )
            saved = st.form_submit_button('Применить историю')
        if saved:
            st.session_state.history = parse_history(history_text)
            st.session_state.pending_history = '\n'.join(st.session_state.history)
            st.session_state.last_result = None
            st.rerun()
        st.button('Очистить историю', on_click=set_history, args=([],))
        with st.expander('Примеры интересов'):
            st.button('Космос', on_click=set_history,
                      args=(['астрономия', 'космические исследования', 'спутники'],))
            st.button('Медицина', on_click=set_history,
                      args=(['медицинские исследования', 'вакцины', 'генетика'],))
            st.button('Экономика', on_click=set_history,
                      args=(['инфляция', 'ключевая ставка', 'банковский сектор'],))
        st.caption('История хранится в текущей сессии приложения. После закрытия сессии её можно задать заново.')

    mode = st.radio(
        'Что показать?', ['По текущему запросу', 'Персональная лента'], horizontal=True
    )
    with st.form('recommendation_form'):
        query = st.text_input(
            'Текущий запрос',
            placeholder='Например: запуски космических ракет',
            disabled=mode == 'Персональная лента',
        )
        left, right = st.columns(2)
        with left:
            method_label = st.selectbox('Метод', ['Семантический — E5', 'Лексический — TF-IDF'])
        with right:
            top_n = st.selectbox('Количество рекомендаций', [5, 10], index=1)
        use_history = st.checkbox('Учитывать историю запросов', value=True,
                                  disabled=mode == 'Персональная лента')
        remember = st.checkbox('Добавлять успешный текущий запрос в историю после выдачи', value=True,
                               disabled=mode == 'Персональная лента')
        submitted = st.form_submit_button('Получить рекомендации', type='primary')

    if submitted:
        current = query.strip() if mode == 'По текущему запросу' else ''
        history = (
            list(st.session_state.history)
            if mode == 'Персональная лента' or use_history
            else []
        )
        if mode == 'По текущему запросу' and not current:
            st.session_state.last_result = None
            st.warning('Введите текущий запрос. Для рекомендаций только по интересам выберите персональную ленту.')
        elif mode == 'Персональная лента' and not history:
            st.session_state.last_result = None
            st.warning('Задайте историю в боковой панели и нажмите «Применить историю».')
        else:
            try:
                with st.spinner('Подбираем новости. Первое обращение к E5 включает загрузку модели…'):
                    engine, positions, lock = load_engine(str(ROOT))
                    method = 'e5' if method_label == 'Семантический — E5' else 'tfidf'
                    with lock:
                        result = engine.recommend(
                            current, history, method=method, alpha=ALPHA, top_n=top_n
                        )
                st.session_state.last_result = result
                # Записываем текущий запрос только после формирования выдачи.
                if current and remember and result['items']:
                    st.session_state.history = (st.session_state.history + [current])[-5:]
                    st.session_state.pending_history = '\n'.join(st.session_state.history)
                    st.rerun()
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                st.session_state.last_result = None
                st.error('Не удалось получить рекомендации. Проверьте подготовленные данные и модели.')
                with st.expander('Подробности ошибки'):
                    st.text(str(exc))

    result = st.session_state.last_result
    if result is None:
        st.info('Введите запрос или задайте интересы для персональной ленты.')
        return
    st.subheader('Рекомендации')
    st.write('Запрос: ' + (result['query'] or 'без текущего запроса'))
    st.write('История, использованная для этой выдачи: ' +
             ('; '.join(result['history_used']) or 'не использовалась'))
    st.caption('Метод: ' + result['method'].upper() +
               f" · Найдено: {len(result['items'])} · Вызов рекомендателя: {result['elapsed_seconds']:.3f} с")
    for message in result['messages']:
        st.warning(message)
    if not result['items']:
        st.info('Рекомендации не найдены. Измените запросы или попробуйте другой метод.')
        return
    engine, positions, lock = load_engine(str(ROOT))
    for item in result['items']:
        with st.container(border=True):
            st.subheader(f"{item['rank']}. {item['title']}")
            st.caption(f"{item['date']} · {item['topic']}")
            article = engine.articles.iloc[positions[item['article_id']]]
            full_text = str(article['text'])
            st.write(full_text[:380] + ('…' if len(full_text) > 380 else ''))
            st.link_button('Открыть на Lenta.ru', item['url'])
            with st.expander('Полный текст'):
                st.text(full_text)
            with st.expander('Оценки ранжирования'):
                st.caption('Сходство используется для порядка выдачи и не является вероятностью релевантности.')
                st.write(f"Итоговая оценка: {item['score']:.4f}")
                if item['query_score'] is not None:
                    st.write(f"Сходство с запросом: {item['query_score']:.4f}")
                if item['history_score'] is not None:
                    st.write(f"Сходство с профилем: {item['history_score']:.4f}")
    with st.expander('Как формируется выдача'):
        st.write('При наличии запроса и истории: 80% сходства с текущим запросом и 20% сходства с профилем. '
                 'Без истории используется только запрос; в персональной ленте — только профиль интересов.')
        st.write('Рубрика и дата показываются для удобства чтения и не участвуют в ранжировании. '
                 'Новый запрос добавляется в историю после формирования рекомендаций, если включено сохранение.')
    st.download_button('Скачать текущую выдачу JSON',
                       json.dumps(result, ensure_ascii=False, indent=2),
                       file_name='recommendations.json', mime='application/json')


if __name__ == '__main__':
    main()
