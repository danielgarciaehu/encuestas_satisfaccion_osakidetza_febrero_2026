import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import re
from pathlib import Path

st.set_page_config(
    page_title="Dashboard Encuestas CCEE",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #1a3a5c; }
    h2, h3 { color: #2c5f8a; }
</style>
""", unsafe_allow_html=True)

CSV_PATH = Path(__file__).parent / "CCEE_Extrahospi_GLOBAL.csv"

COLS = [
    "Servicio", "ID", "FechaEnvio", "UltimaPagina", "Idioma", "Semilla",
    "FechaInicio", "FechaUltimaAccion", "_vacio",
    "Edad", "Genero", "Ambulatorio",
    "_sec1",
    "Acc_EsperaFirstVisit",
    "SegundaVisita",
    "Acc_EsperaSeguimiento",
    "Acc_EsperaSalaEspera",
    "Info_ClaridadInfo",
    "Info_PosibilidadPreguntar",
    "_sec2",
    "Hum_Trato",
    "Hum_Explicaciones",
    "Hum_DuracionConsultas",
    "Hum_ConfortInstalaciones",
    "Coord_Coordinacion",
    "Euskera",
    "Res_MejoraSalud",
    "Seg_Incidente",
    "Seg_DescripcionIncidente",
    "Val_SatisfaccionGlobal",
    "Val_Recomendaria",
    "Val_Comentarios",
]

DIMENSIONES = {
    "Accesibilidad": ["Acc_EsperaFirstVisit", "Acc_EsperaSeguimiento", "Acc_EsperaSalaEspera"],
    "Información": ["Info_ClaridadInfo", "Info_PosibilidadPreguntar"],
    "Humanización": ["Hum_Trato", "Hum_Explicaciones", "Hum_DuracionConsultas", "Hum_ConfortInstalaciones"],
    "Coordinación": ["Coord_Coordinacion"],
    "Resultados": ["Res_MejoraSalud"],
    "Global": ["Val_SatisfaccionGlobal"],
}

LABELS = {
    "Acc_EsperaFirstVisit": "Espera 1ª visita",
    "Acc_EsperaSeguimiento": "Espera seguimiento",
    "Acc_EsperaSalaEspera": "Espera en sala",
    "Info_ClaridadInfo": "Claridad información",
    "Info_PosibilidadPreguntar": "Preguntar dudas",
    "Hum_Trato": "Trato personal",
    "Hum_Explicaciones": "Explicaciones recuperación",
    "Hum_DuracionConsultas": "Duración consultas",
    "Hum_ConfortInstalaciones": "Confort instalaciones",
    "Coord_Coordinacion": "Coordinación sanitaria",
    "Euskera": "Uso euskera",
    "Res_MejoraSalud": "Mejora salud",
    "Val_SatisfaccionGlobal": "Satisfacción global",
}

NUMERIC_COLS = list(LABELS.keys())

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, sep=";", header=0, names=COLS, encoding="latin-1", low_memory=False)
    df = df[df["Servicio"].notna() & ~df["Servicio"].str.contains(r"Servicio|^$", na=True)]
    df = df[~df["Servicio"].str.len().gt(50)]  # drop comment rows bleed into col 1

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Euskera"] = df["Euskera"].astype(str).replace("No procede", np.nan).replace("nan", np.nan)
    df["Euskera"] = pd.to_numeric(df["Euskera"], errors="coerce")

    df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce")
    df["GrupoEdad"] = pd.cut(
        df["Edad"],
        bins=[0, 30, 45, 60, 75, 120],
        labels=["<30", "30-44", "45-59", "60-74", "75+"],
    )

    df["Recomendaria_bool"] = df["Val_Recomendaria"].astype(str).str.strip().str.lower().map({"sí": True, "si": True, "no": False})
    df["Incidente_bool"] = df["Seg_Incidente"].astype(str).str.strip().str.lower().map({"sí": True, "si": True, "no": False})

    df["FechaEnvio"] = pd.to_datetime(df["FechaEnvio"], format="%d/%m/%Y %H:%M", errors="coerce")
    df["Mes"] = df["FechaEnvio"].dt.to_period("M").astype(str)

    for dim, cols in DIMENSIONES.items():
        available = [c for c in cols if c in df.columns]
        df[f"Dim_{dim}"] = df[available].mean(axis=1)

    return df

df_raw = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Osakidetza_logo.svg/320px-Osakidetza_logo.svg.png", width=160)
st.sidebar.title("Filtros")

servicios = sorted(df_raw["Servicio"].dropna().unique())
sel_servicios = st.sidebar.multiselect("Servicio", servicios, default=servicios)

ambulatorios = sorted(df_raw["Ambulatorio"].dropna().unique())
sel_ambulatorios = st.sidebar.multiselect("Ambulatorio", ambulatorios, default=ambulatorios)

generos = sorted(df_raw["Genero"].dropna().unique())
sel_generos = st.sidebar.multiselect("Género", generos, default=generos)

edad_min, edad_max = int(df_raw["Edad"].min(skipna=True)), int(df_raw["Edad"].max(skipna=True))
sel_edad = st.sidebar.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max))

solo_completas = st.sidebar.checkbox("Solo respuestas completas (pág. 4)", value=False)

df = df_raw.copy()
if sel_servicios:
    df = df[df["Servicio"].isin(sel_servicios)]
if sel_ambulatorios:
    df = df[df["Ambulatorio"].isin(sel_ambulatorios)]
if sel_generos:
    df = df[df["Genero"].isin(sel_generos)]
df = df[df["Edad"].isna() | df["Edad"].between(sel_edad[0], sel_edad[1])]
if solo_completas:
    df = df[df["UltimaPagina"].astype(str).str.strip().str.startswith("4")]

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏥 Dashboard Encuestas CCEE Extrahospitalaria")
st.caption(f"Datos de {df['FechaEnvio'].min().strftime('%d/%m/%Y') if df['FechaEnvio'].notna().any() else '—'} "
           f"a {df['FechaEnvio'].max().strftime('%d/%m/%Y') if df['FechaEnvio'].notna().any() else '—'} · "
           f"{len(df):,} respuestas seleccionadas de {len(df_raw):,} totales")

tabs = st.tabs(["📊 Visión General", "🏷️ Por Servicio", "🏢 Por Ambulatorio", "👥 Demografía", "⏱️ Accesibilidad", "💬 Comentarios"])

# ── TAB 1: VISIÓN GENERAL ────────────────────────────────────────────────────
with tabs[0]:
    n_resp = len(df)
    n_con_datos = df["Val_SatisfaccionGlobal"].notna().sum()
    media_global = df["Val_SatisfaccionGlobal"].mean()
    pct_rec = df["Recomendaria_bool"].mean() * 100 if df["Recomendaria_bool"].notna().any() else None
    pct_inc = df["Incidente_bool"].mean() * 100 if df["Incidente_bool"].notna().any() else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respuestas totales", f"{n_resp:,}")
    c2.metric("Con puntuación global", f"{n_con_datos:,}")
    c3.metric("Satisfacción global media", f"{media_global:.2f} / 10" if not np.isnan(media_global) else "—")
    c4.metric("Recomendaría (%)", f"{pct_rec:.1f}%" if pct_rec is not None else "—")
    c5.metric("Incidentes seguridad (%)", f"{pct_inc:.1f}%" if pct_inc is not None else "—")

    st.divider()
    col_radar, col_hist = st.columns([1, 1])

    with col_radar:
        st.subheader("Puntuación media por dimensión")
        dim_means = {dim: df[f"Dim_{dim}"].mean() for dim in DIMENSIONES if f"Dim_{dim}" in df.columns}
        dims = list(dim_means.keys())
        vals = [round(v, 2) if not np.isnan(v) else 0 for v in dim_means.values()]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dims + [dims[0]],
            fill="toself",
            fillcolor="rgba(41,128,185,0.2)",
            line=dict(color="#2980b9", width=2),
            marker=dict(size=6),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            margin=dict(l=60, r=60, t=30, b=30),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_hist:
        st.subheader("Distribución satisfacción global")
        fig_hist = px.histogram(
            df.dropna(subset=["Val_SatisfaccionGlobal"]),
            x="Val_SatisfaccionGlobal",
            nbins=11,
            color_discrete_sequence=["#2980b9"],
            labels={"Val_SatisfaccionGlobal": "Puntuación (0-10)"},
        )
        fig_hist.update_layout(bargap=0.08, margin=dict(t=30, b=30), height=380, yaxis_title="Nº respuestas")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Evolución mensual – Satisfacción global media")
    evol = df.dropna(subset=["Val_SatisfaccionGlobal", "Mes"]).groupby("Mes")["Val_SatisfaccionGlobal"].agg(["mean", "count"]).reset_index()
    evol.columns = ["Mes", "Media", "N"]
    evol = evol[evol["N"] >= 5]
    if not evol.empty:
        fig_evol = px.line(evol, x="Mes", y="Media", markers=True,
                           labels={"Media": "Puntuación media", "Mes": ""},
                           color_discrete_sequence=["#1a6e9e"])
        fig_evol.update_layout(yaxis_range=[0, 10], margin=dict(t=20, b=20), height=300)
        st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.info("No hay suficientes datos con fecha para mostrar evolución.")

# ── TAB 2: POR SERVICIO ───────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Satisfacción global media por servicio")
    srv_global = df.groupby("Servicio")["Val_SatisfaccionGlobal"].agg(["mean", "count"]).reset_index()
    srv_global.columns = ["Servicio", "Media", "N"]
    srv_global = srv_global[srv_global["N"] >= 5].sort_values("Media", ascending=True)

    fig_srv = px.bar(
        srv_global, x="Media", y="Servicio", orientation="h",
        color="Media", color_continuous_scale="RdYlGn", range_color=[0, 10],
        text=srv_global["Media"].round(2).astype(str) + " (" + srv_global["N"].astype(str) + ")",
        labels={"Media": "Puntuación media", "Servicio": ""},
    )
    fig_srv.update_traces(textposition="outside")
    fig_srv.update_layout(height=400, margin=dict(t=20, b=20), coloraxis_showscale=False, xaxis_range=[0, 11])
    st.plotly_chart(fig_srv, use_container_width=True)

    st.subheader("Heatmap – Dimensiones por servicio")
    dim_cols = [f"Dim_{d}" for d in DIMENSIONES if f"Dim_{d}" in df.columns]
    hm_data = df.groupby("Servicio")[dim_cols].mean().round(2)
    hm_data.columns = list(DIMENSIONES.keys())
    hm_data = hm_data.dropna(how="all")

    fig_hm = px.imshow(
        hm_data,
        text_auto=True, aspect="auto",
        color_continuous_scale="RdYlGn", zmin=0, zmax=10,
        labels=dict(color="Puntuación"),
    )
    fig_hm.update_layout(height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_hm, use_container_width=True)

    st.subheader("% Recomendaría por servicio")
    rec_srv = df.groupby("Servicio")["Recomendaria_bool"].agg(lambda x: x.mean() * 100 if x.notna().any() else np.nan).reset_index()
    rec_srv.columns = ["Servicio", "PctRec"]
    rec_srv = rec_srv.dropna().sort_values("PctRec", ascending=True)
    fig_rec = px.bar(rec_srv, x="PctRec", y="Servicio", orientation="h",
                     color="PctRec", color_continuous_scale="RdYlGn", range_color=[0, 100],
                     labels={"PctRec": "% Recomendaría", "Servicio": ""},
                     text=rec_srv["PctRec"].round(1).astype(str) + "%")
    fig_rec.update_traces(textposition="outside")
    fig_rec.update_layout(height=380, margin=dict(t=20, b=20), coloraxis_showscale=False, xaxis_range=[0, 115])
    st.plotly_chart(fig_rec, use_container_width=True)

# ── TAB 3: POR AMBULATORIO ───────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Satisfacción global media por ambulatorio")
    amb_global = df.groupby("Ambulatorio")["Val_SatisfaccionGlobal"].agg(["mean", "count"]).reset_index()
    amb_global.columns = ["Ambulatorio", "Media", "N"]
    amb_global = amb_global[amb_global["N"] >= 5].sort_values("Media", ascending=True)

    fig_amb = px.bar(
        amb_global, x="Media", y="Ambulatorio", orientation="h",
        color="Media", color_continuous_scale="RdYlGn", range_color=[0, 10],
        text=amb_global["Media"].round(2).astype(str) + " (" + amb_global["N"].astype(str) + ")",
        labels={"Media": "Puntuación media", "Ambulatorio": ""},
    )
    fig_amb.update_traces(textposition="outside")
    fig_amb.update_layout(height=400, margin=dict(t=20, b=20), coloraxis_showscale=False, xaxis_range=[0, 11])
    st.plotly_chart(fig_amb, use_container_width=True)

    st.subheader("Heatmap – Dimensiones por ambulatorio")
    dim_cols = [f"Dim_{d}" for d in DIMENSIONES if f"Dim_{d}" in df.columns]
    hm_amb = df.groupby("Ambulatorio")[dim_cols].mean().round(2)
    hm_amb.columns = list(DIMENSIONES.keys())
    hm_amb = hm_amb.dropna(how="all")

    fig_hm_amb = px.imshow(
        hm_amb, text_auto=True, aspect="auto",
        color_continuous_scale="RdYlGn", zmin=0, zmax=10,
        labels=dict(color="Puntuación"),
    )
    fig_hm_amb.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig_hm_amb, use_container_width=True)

    st.subheader("Cruce servicio × ambulatorio – Satisfacción global")
    pivot = df.pivot_table(values="Val_SatisfaccionGlobal", index="Servicio", columns="Ambulatorio", aggfunc="mean").round(2)
    if not pivot.empty:
        fig_cross = px.imshow(
            pivot, text_auto=True, aspect="auto",
            color_continuous_scale="RdYlGn", zmin=0, zmax=10,
            labels=dict(color="Puntuación"),
        )
        fig_cross.update_layout(height=450, margin=dict(t=20, b=20))
        st.plotly_chart(fig_cross, use_container_width=True)

# ── TAB 4: DEMOGRAFÍA ────────────────────────────────────────────────────────
with tabs[3]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución por género")
        gen_counts = df["Genero"].value_counts().reset_index()
        gen_counts.columns = ["Género", "N"]
        fig_gen = px.pie(gen_counts, names="Género", values="N",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         hole=0.4)
        fig_gen.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_gen, use_container_width=True)

    with col2:
        st.subheader("Distribución por grupo de edad")
        edad_counts = df["GrupoEdad"].value_counts().sort_index().reset_index()
        edad_counts.columns = ["Grupo", "N"]
        fig_edad = px.bar(edad_counts, x="Grupo", y="N",
                          color_discrete_sequence=["#2980b9"],
                          labels={"Grupo": "Grupo de edad", "N": "Nº respuestas"})
        fig_edad.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_edad, use_container_width=True)

    st.subheader("Satisfacción global por género")
    gen_sat = df.groupby("Genero")["Val_SatisfaccionGlobal"].agg(["mean", "count"]).reset_index()
    gen_sat.columns = ["Género", "Media", "N"]
    gen_sat = gen_sat[gen_sat["N"] >= 5]
    fig_gsat = px.bar(gen_sat, x="Género", y="Media",
                      color="Media", color_continuous_scale="RdYlGn", range_color=[0, 10],
                      text=gen_sat["Media"].round(2), labels={"Media": "Puntuación media"})
    fig_gsat.update_traces(textposition="outside")
    fig_gsat.update_layout(height=320, yaxis_range=[0, 11], coloraxis_showscale=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_gsat, use_container_width=True)

    st.subheader("Satisfacción global por grupo de edad")
    edad_sat = df.groupby("GrupoEdad", observed=True)["Val_SatisfaccionGlobal"].agg(["mean", "count"]).reset_index()
    edad_sat.columns = ["GrupoEdad", "Media", "N"]
    edad_sat = edad_sat[edad_sat["N"] >= 5]
    fig_esat = px.bar(edad_sat, x="GrupoEdad", y="Media",
                      color="Media", color_continuous_scale="RdYlGn", range_color=[0, 10],
                      text=edad_sat["Media"].round(2), labels={"GrupoEdad": "Grupo de edad", "Media": "Puntuación media"})
    fig_esat.update_traces(textposition="outside")
    fig_esat.update_layout(height=320, yaxis_range=[0, 11], coloraxis_showscale=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_esat, use_container_width=True)

# ── TAB 5: ACCESIBILIDAD ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Tiempos de espera – Puntuación media por servicio")

    acc_cols = {
        "Acc_EsperaFirstVisit": "Espera 1ª visita",
        "Acc_EsperaSeguimiento": "Espera seguimiento",
        "Acc_EsperaSalaEspera": "Espera en sala",
    }
    acc_data = df.groupby("Servicio")[list(acc_cols.keys())].mean().round(2)
    acc_data = acc_data.rename(columns=acc_cols).reset_index()
    acc_melted = acc_data.melt(id_vars="Servicio", var_name="Indicador", value_name="Puntuación")

    fig_acc = px.bar(
        acc_melted, x="Servicio", y="Puntuación", color="Indicador",
        barmode="group", color_discrete_sequence=["#2980b9", "#e67e22", "#27ae60"],
        labels={"Puntuación": "Puntuación media (0-10)"},
    )
    fig_acc.update_layout(height=420, margin=dict(t=20, b=20), yaxis_range=[0, 11])
    st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader("¿Segunda visita de seguimiento?")
    seg = df["SegundaVisita"].astype(str).str.strip().replace("nan", np.nan).dropna().value_counts().reset_index()
    seg.columns = ["Respuesta", "N"]
    fig_seg = px.pie(seg, names="Respuesta", values="N", hole=0.4,
                     color_discrete_sequence=["#27ae60", "#e74c3c", "#95a5a6"])
    fig_seg.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Espera 1ª visita vs. espera en sala – por servicio")
    scatter_df = df.dropna(subset=["Acc_EsperaFirstVisit", "Acc_EsperaSalaEspera", "Servicio"])
    fig_sc = px.scatter(
        scatter_df, x="Acc_EsperaFirstVisit", y="Acc_EsperaSalaEspera",
        color="Servicio", opacity=0.5, size_max=6,
        labels={"Acc_EsperaFirstVisit": "Espera 1ª visita", "Acc_EsperaSalaEspera": "Espera en sala"},
    )
    fig_sc.update_layout(height=420, margin=dict(t=20, b=20), xaxis_range=[-0.5, 10.5], yaxis_range=[-0.5, 10.5])
    st.plotly_chart(fig_sc, use_container_width=True)

# ── TAB 6: COMENTARIOS ───────────────────────────────────────────────────────
with tabs[5]:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Incidentes de seguridad reportados")
        inc_total = df["Incidente_bool"].notna().sum()
        inc_si = df["Incidente_bool"].sum()
        inc_no = (df["Incidente_bool"] == False).sum()
        st.metric("Total responden", f"{int(inc_total):,}")
        st.metric("Reportan incidente", f"{int(inc_si):,} ({inc_si/inc_total*100:.1f}%)" if inc_total else "—")
        st.metric("Sin incidente", f"{int(inc_no):,}")

        inc_srv = df.groupby("Servicio")["Incidente_bool"].mean().mul(100).round(1).reset_index()
        inc_srv.columns = ["Servicio", "% Incidentes"]
        inc_srv = inc_srv.dropna().sort_values("% Incidentes", ascending=False)
        fig_inc = px.bar(inc_srv, x="Servicio", y="% Incidentes",
                         color="% Incidentes", color_continuous_scale="Reds",
                         labels={"% Incidentes": "% pacientes con incidente"})
        fig_inc.update_layout(height=350, margin=dict(t=20, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_inc, use_container_width=True)

    with col_b:
        st.subheader("Descripción de incidentes")
        incidentes = df[df["Incidente_bool"] == True][["Servicio", "Ambulatorio", "Seg_DescripcionIncidente"]].dropna(subset=["Seg_DescripcionIncidente"])
        incidentes = incidentes[incidentes["Seg_DescripcionIncidente"].astype(str).str.strip() != ""]
        st.dataframe(incidentes.rename(columns={"Seg_DescripcionIncidente": "Descripción"}), height=400, use_container_width=True)

    st.divider()
    st.subheader("Comentarios libres")
    comentarios = df[["Servicio", "Ambulatorio", "Genero", "Edad", "Val_SatisfaccionGlobal", "Val_Comentarios"]].dropna(subset=["Val_Comentarios"])
    comentarios = comentarios[comentarios["Val_Comentarios"].astype(str).str.strip() != ""]

    buscar = st.text_input("Buscar en comentarios", placeholder="escribe para filtrar...")
    if buscar:
        comentarios = comentarios[comentarios["Val_Comentarios"].str.contains(buscar, case=False, na=False)]

    st.caption(f"{len(comentarios):,} comentarios")
    st.dataframe(
        comentarios.rename(columns={
            "Val_SatisfaccionGlobal": "Nota global",
            "Val_Comentarios": "Comentario",
        }),
        height=400,
        use_container_width=True,
    )
