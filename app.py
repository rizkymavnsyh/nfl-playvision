"""Streamlit wrapper for the fully offline HTML replay."""
from pathlib import Path
import json,pandas as pd
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title='NFL PlayVision',page_icon='🏈',layout='wide')
root=Path(__file__).resolve().parent
st.title('NFL PlayVision')
st.caption('21 replay clips from 13 games • 0.5 / 1.0-second movement forecasts • Receiver–defender geometry')
if not (root/'PlayVision_Offline.html').exists():st.info('Run src/export_demo.py after training.');st.stop()
components.html((root/'PlayVision_Offline.html').read_text(),height=1080,scrolling=True)
with st.expander('Evaluation, scope, and data'):
    metrics=json.loads((root/'assets/replay_data.json').read_text())['metrics'];st.json(metrics)
    st.write('Scored on two 2021 games. The complete 2017 replay is an external-era illustration, not the scored test. These percentages measure position error tolerance, not game-outcome accuracy.')
    st.dataframe(pd.read_csv(root/'reports/position_metrics.csv'),hide_index=True)
st.download_button('Download offline replay',(root/'PlayVision_Offline.html').read_bytes(),'PlayVision_Offline.html','text/html')
