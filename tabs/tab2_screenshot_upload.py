import streamlit as st
from datetime import date
from PIL import Image

from utils.openai_vision import detect_daily_sku_rows_from_long_screenshot
from utils.cleaning import (
    clean_daily_sku_sales_df,
    dataframe_to_supabase_rows,
)
from utils.supabase_client import insert_rows, upsert_rows


def render_tab2_screenshot_upload():
    st.header("Tab 2: Upload Daily SKU Screenshot")
    st.caption(
        "Upload a screenshot image. OpenAI will detect the daily SKU sales data, "
        "then you can review and upload it into `daily_sku_sales`."
    )

    sale_date = st.date_input(
        "Sales date for this screenshot",
        value=date.today(),
        key="daily_sku_sale_date",
    )

    # Image upload only
    uploaded_image = st.file_uploader(
        "Upload screenshot image",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="daily_sku_screenshot_image",
    )

    upload_mode = st.radio(
        "Upload mode",
        options=[
            "Insert new rows",
            "Upsert / replace duplicate rows",
        ],
        index=1,
        key="daily_sku_upload_mode",
    )

    if uploaded_image is None:
        st.info("Upload a screenshot image to start.")
        return

    # Validate image
    try:
        image = Image.open(uploaded_image)
        st.image(
            image,
            caption=f"Uploaded image: {uploaded_image.name}",
            use_container_width=True,
        )

        # Reset pointer so OpenAI function can read the uploaded file again
        uploaded_image.seek(0)

    except Exception as e:
        st.error("The uploaded file is not a valid image.")
        st.exception(e)
        return

    if st.button("Detect data with OpenAI", key="detect_daily_sku_btn"):
        try:
            uploaded_image.seek(0)

            detected_df = detect_daily_sku_rows_from_long_screenshot(uploaded_image)

            if detected_df.empty:
                st.warning(
                    "No rows detected. Try uploading a clearer screenshot or cropping closer to the table."
                )
                return

            cleaned_df = clean_daily_sku_sales_df(
                detected_df,
                sale_date=sale_date,
                source_file=uploaded_image.name,
            )

            st.session_state["daily_sku_detected_df"] = cleaned_df
            st.success(f"Detected {len(cleaned_df)} valid rows.")

        except Exception as e:
            st.error("OpenAI image detection failed.")
            st.exception(e)

    if "daily_sku_detected_df" not in st.session_state:
        return

    st.subheader("Review and edit detected data")

    edited_df = st.data_editor(
        st.session_state["daily_sku_detected_df"],
        use_container_width=True,
        num_rows="dynamic",
        key="daily_sku_detected_editor",
    )

    st.subheader("Final upload preview")
    st.dataframe(edited_df, use_container_width=True)

    if st.button("Confirm and upload to daily_sku_sales", key="upload_daily_sku_confirm"):
        try:
            rows = dataframe_to_supabase_rows(edited_df)

            table_name = st.secrets.get(
                "DAILY_SKU_SALES_TABLE",
                "daily_sku_sales",
            )

            if upload_mode == "Upsert / replace duplicate rows":
                response = upsert_rows(
                    table_name=table_name,
                    rows=rows,
                    on_conflict="sale_date,sku_no",
                )
            else:
                response = insert_rows(
                    table_name=table_name,
                    rows=rows,
                )

            st.success(f"Uploaded {len(rows)} rows to `{table_name}`.")
            st.write(response)

        except Exception as e:
            st.error("Failed to upload daily SKU sales.")
            st.exception(e)
