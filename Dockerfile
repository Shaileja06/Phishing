FROM python:3.11.4
COPY . /phishing
WORKDIR /phishing
RUN pip install -r requirements.txt
EXPOSE 80
CMD streamlit run app.py  