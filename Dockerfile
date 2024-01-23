FROM python
COPY . /phishing
WORKDIR /phishing
RUN pip install -r requirements.txt
CMD streamlit run app.py  