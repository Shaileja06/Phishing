import re
import socket
from urllib.parse import urlparse , parse_qs
import joblib
import numpy as np

def extract_additional_url_features(url):
    parsed_url = urlparse(url)

    # Extracting features based on characters in the URL
    char_count_features = {
        'Unnamed':0,
        'qty_dot_url': url.count('.'),
        'qty_hyphen_url': url.count('-'),
        'qty_underline_url': url.count('_'),
        'qty_slash_url': url.count('/'),
        'qty_questionmark_url': url.count('?'),
        'qty_equal_url': url.count('='),
        'qty_at_url': url.count('@'),
        'qty_exclamation_url': url.count('!'),
        'qty_space_url': url.count(' '),
        'qty_tilde_url': url.count('~'),
        'qty_comma_url': url.count(','),
        'qty_plus_url': url.count('+'),
        'qty_asterisk_url': url.count('*'),
        'qty_hashtag_url': url.count('#'),
        'qty_dollar_url': url.count('$'),
        'qty_percent_url': url.count('%'),
        'qty_tld_url': url.count('.') - 1,  # Subtracting 1 to exclude the dot in TLD
        'length_url': len(url)
    }

    return char_count_features


def extract_additional_domain_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Extracting features based on characters in the domain
    char_count_features = {
        'qty_dot_domain': domain.count('.'),
        'qty_hyphen_domain': domain.count('-'),
        'qty_underline_domain': domain.count('_'),
        'qty_at_domain': domain.count('@'),
        'qty_vowels_domain': sum(1 for char in domain if char.lower() in 'aeiou'),
    }

    # Check if the domain is an IP address
    try:
        ip_address = socket.gethostbyname(domain)
        domain_in_ip = 1
    except socket.error:
        domain_in_ip = 0

    # Check if "server" or "client" is present as a separate word in the domain
    server_client_domain = 1 if re.search(r'\b(server|client)\b', domain, flags=re.IGNORECASE) else 0

    char_count_features['domain_in_ip'] = domain_in_ip
    char_count_features['server_client_domain'] = server_client_domain

    return char_count_features

def extract_additional_path_features(url):
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Extracting features based on characters in the directory
    directory_features = {
        'qty_dot_directory': path.count('.'),
        'qty_hyphen_directory': path.count('-'),
        'qty_underline_directory': path.count('_'),
        'qty_questionmark_directory': path.count('?'),
        'directory_length': len(path),
    }

    # Extracting features based on characters in the file
    file_features = {
        'qty_hyphen_file': parsed_url.path.rfind('-'),
        'file_length': len(parsed_url.path),
    }

    return {**directory_features, **file_features}

def extract_additional_params_features(url):
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)

    # Extracting features based on characters in the parameters
    params_features = {
        'qty_dot_params': sum(value[0].count('.') for value in params.values()),
        'qty_hyphen_params': sum(value[0].count('-') for value in params.values()),
        'qty_underline_params': sum(value[0].count('_') for value in params.values()),
        'qty_slash_params': sum(value[0].count('/') for value in params.values()),
        'qty_questionmark_params': sum(value[0].count('?') for value in params.values()),
        'qty_percent_params': sum(value[0].count('%') for value in params.values()),
    }

    return params_features

def email_in_url(url):
    return 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', url) else 0

def url_shortened(url):
    return 1 if len(url) <= 25 else 0

def extract_all_features(url):
    parsed_url = urlparse(url)
    # Extract URL-based features
    url_features = extract_additional_url_features(url)

    # Extract Domain-based features
    domain_features = extract_additional_domain_features(url)

    # Extract Page-based features
    path_features = extract_additional_path_features(url)

    # Extract Params-based features
    params_features = extract_additional_params_features(url)

    # Extract Additional Features
    additional_features = {
        'email_in_url': email_in_url(url),
        #'time_domain_activation': time_domain_activation(parsed_url.netloc),
        #'time_domain_expiration': time_domain_expiration(parsed_url.netloc),
        'url_shortened': url_shortened(url),
    }

    # Combine all features
    all_features = {**url_features, **domain_features, **path_features, **params_features, **additional_features}

    return all_features


def prediction(extracted_features):
    #loaded_pipeline = joblib.load('/content/tpot_xgbclassifier_pipeline.joblib')

    data = np.array(list(extracted_features.values())).reshape(1, -1)

    # Assuming you have a scaler object
    scaler = joblib.load('artifacts\components\standard.joblib')
    scaled_data = scaler.transform(data)

    # Assuming you have a PCA object
    pca = joblib.load('artifacts\components\pca.joblib')
    pca_transformed_data = pca.transform(scaled_data)

    # Use the trained XGBBoost for prediction
    #prediction = loaded_pipeline.predict(pca_transformed_data)
    tpot = joblib.load('artifacts\model\model.joblib')
    prediction = tpot.predict(pca_transformed_data)

    print(prediction)
    return prediction