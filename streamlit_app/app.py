"""A polished, non-technical Streamlit interface for local house-price scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import HouseFeatures
from src.prediction import ModelNotAvailableError, predict_price


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


def _selectbox(label: str, options: list[str], default: str) -> str:
    """Render a categorical input with a schema-aligned default."""
    return st.selectbox(label, options, index=options.index(default))


def collect_features() -> HouseFeatures:
    """Render concise property inputs and build the validated request object."""
    st.sidebar.header("Property details")
    st.sidebar.caption("Values are used only to estimate this property's sale price.")
    with st.sidebar:
        st.subheader("Size and layout")
        lot_area = st.number_input("Lot area (sq ft)", 1, 500_000, 8_450)
        living_area = st.number_input("Above-ground living area (sq ft)", 1, 20_000, 1_710)
        basement_area = st.number_input("Basement area (sq ft)", 0, 20_000, 856)
        first_floor_area = st.number_input("First-floor area (sq ft)", 1, 20_000, 856)
        second_floor_area = st.number_input("Second-floor area (sq ft)", 0, 20_000, 854)
        bedrooms = st.number_input("Bedrooms above ground", 0, 20, 3)
        rooms = st.number_input("Total rooms above ground", 1, 30, 8)
        st.subheader("Quality and age")
        overall_quality = st.slider("Overall quality (1 = low, 10 = excellent)", 1, 10, 7)
        overall_condition = st.slider("Overall condition (1 = poor, 10 = excellent)", 1, 10, 5)
        year_built = st.number_input("Year built", 1800, 2100, 2003)
        year_remodeled = st.number_input("Year remodeled", 1800, 2100, 2003)
        year_sold = st.number_input("Year sold", 2006, 2100, 2008)

    left, right = st.columns(2)
    with left:
        st.subheader("Amenities")
        full_baths = st.number_input("Full bathrooms", 0, 10, 2)
        half_baths = st.number_input("Half bathrooms", 0, 10, 1)
        basement_full_baths = st.number_input("Basement full bathrooms", 0, 10, 1)
        basement_half_baths = st.number_input("Basement half bathrooms", 0, 10, 0)
        garage_cars = st.number_input("Garage capacity (cars)", 0, 8, 2)
        garage_area = st.number_input("Garage area (sq ft)", 0, 5_000, 548)
        deck_area = st.number_input("Wood deck area (sq ft)", 0, 10_000, 0)
        porch_area = st.number_input("Open porch area (sq ft)", 0, 10_000, 61)
    with right:
        st.subheader("Property type")
        zoning = _selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"], "RL")
        neighborhood = _selectbox(
            "Neighborhood",
            ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NridgHt", "OldTown", "Edwards", "NAmes"],
            "CollgCr",
        )
        building_type = _selectbox("Building type", ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"], "1Fam")
        house_style = _selectbox("House style", ["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"], "2Story")
        exterior_quality = _selectbox("Exterior quality", ["Ex", "Gd", "TA", "Fa", "Po"], "Gd")
        kitchen_quality = _selectbox("Kitchen quality", ["Ex", "Gd", "TA", "Fa", "Po"], "Gd")
        heating_quality = _selectbox("Heating quality", ["Ex", "Gd", "TA", "Fa", "Po"], "Ex")
        basement_quality = _selectbox("Basement quality", ["Ex", "Gd", "TA", "Fa", "Po", "NoBasement"], "Gd")
        garage_finish = _selectbox("Garage finish", ["Fin", "RFn", "Unf", "NoGarage"], "RFn")
        sale_condition = _selectbox("Sale condition", ["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"], "Normal")
        central_air = _selectbox("Central air", ["Y", "N"], "Y")
        lot_shape = _selectbox("Lot shape", ["Reg", "IR1", "IR2", "IR3"], "Reg")

    return HouseFeatures(
        OverallQual=overall_quality,
        OverallCond=overall_condition,
        GrLivArea=living_area,
        TotalBsmtSF=basement_area,
        **{"1stFlrSF": first_floor_area, "2ndFlrSF": second_floor_area},
        FullBath=full_baths,
        HalfBath=half_baths,
        BsmtFullBath=basement_full_baths,
        BsmtHalfBath=basement_half_baths,
        BedroomAbvGr=bedrooms,
        TotRmsAbvGrd=rooms,
        YearBuilt=year_built,
        YearRemodAdd=year_remodeled,
        YrSold=year_sold,
        GarageCars=garage_cars,
        GarageArea=garage_area,
        WoodDeckSF=deck_area,
        OpenPorchSF=porch_area,
        LotArea=lot_area,
        MSZoning=zoning,
        Neighborhood=neighborhood,
        BldgType=building_type,
        HouseStyle=house_style,
        ExterQual=exterior_quality,
        KitchenQual=kitchen_quality,
        HeatingQC=heating_quality,
        BsmtQual=basement_quality,
        GarageFinish=garage_finish,
        SaleCondition=sale_condition,
        CentralAir=central_air,
        LotShape=lot_shape,
    )


def main() -> None:
    """Run the interactive Streamlit page."""
    st.title("House Price Predictor")
    st.write(
        "Estimate a likely sale price from property size, condition, amenities, and location."
    )
    st.info("This estimate supports comparison and planning; it is not a formal valuation.")
    features = collect_features()
    if st.button("Estimate sale price", type="primary", use_container_width=True):
        try:
            prediction = predict_price(features.to_frame())
        except ModelNotAvailableError:
            st.error("The prediction model is not available yet. Train it before starting the app.")
            return
        except ValueError:
            st.error("We could not evaluate those property details. Please review the inputs.")
            return
        st.success(f"Estimated sale price: **${prediction:,.0f}**")
        st.caption("The estimate is generated by the trained Ames Housing regression pipeline.")


if __name__ == "__main__":
    main()
