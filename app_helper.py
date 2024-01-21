'''import re
import socket
from urllib.parse import urlparse , parse_qs
import joblib
import numpy as np

def extract_additional_url_features(url):
    parsed_url = urlparse(url)

    # Extracting features based on characters in the URL
    char_count_features = {
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
        'qty_tld_url': len(parsed_url.netloc.split('.')[-1]), 
        'length_url': len(url)
    }

    return char_count_features


def extract_additional_domain_features(url):
    # Parse the URL to get the domain
    domain = urlparse(url).netloc

    # Attribute 1: Number of "." signs in the domain
    qty_dot_domain = domain.count('.')

    # Attribute 2: Number of "-" signs in the domain
    qty_hyphen_domain = domain.count('-')

    # Attribute 3: Number of "_" signs in the domain
    qty_underline_domain = domain.count('_')

    # Attribute 4: Number of "@" signs in the domain
    qty_at_domain = domain.count('@')

    # Attribute 5: Number of vowels in the domain
    qty_vowels_domain = sum(1 for char in domain if char.lower() in "aeiou")

    # Attribute 6: Check if the domain is an IP address
    try:
        socket.inet_aton(domain)
        domain_in_ip = 1
    except socket.error:
        domain_in_ip = 0

    # Attribute 7: Server-client domain (assuming "www" is client-side)
    server_client_domain = 1 if domain.startswith("www.") else 0

    # Return the extracted features
    return {
        'qty_dot_domain': qty_dot_domain,
        'qty_hyphen_domain': qty_hyphen_domain,
        'qty_underline_domain': qty_underline_domain,
        'qty_at_domain': qty_at_domain,
        'qty_vowels_domain': qty_vowels_domain,
        'domain_in_ip': domain_in_ip,
        'server_client_domain': server_client_domain
    }

def extract_additional_path_features(url):
    # Parse the URL to get the path
    path = urlparse(url).path

    # Attribute 1: Number of "." signs in the directory
    qty_dot_directory = path.count('.')

    # Attribute 2: Number of "-" signs in the directory
    qty_hyphen_directory = path.count('-')

    # Attribute 3: Number of "_" signs in the directory
    qty_underline_directory = path.count('_')

    # Attribute 4: Number of "%" signs in the directory
    qty_percent_directory = path.count('%')

    # Attribute 5: Length of the directory path
    directory_length = len(path)

    # Return the extracted features
    return {
        'qty_dot_directory': qty_dot_directory,
        'qty_hyphen_directory': qty_hyphen_directory,
        'qty_underline_directory': qty_underline_directory,
        'qty_percent_directory': qty_percent_directory,
        'directory_length': directory_length
    }

def extract_file_features(url):
    # Parse the URL to get the path
    path = urlparse(url).path

    # Extract the file name from the path
    file_name = path.split('/')[-1]

    # Attribute: Length of the file name
    file_length = len(file_name)

    # Return the extracted features
    return {
        'file_length': file_length
    }

def extract_additional_params_features(url):
    # Parse the URL to get the query parameters
    query_params = urlparse(url).query

    # Extract parameter names from the query string
    param_names = parse_qs(query_params).keys()

    # Attribute 1: Number of "." signs in the parameters
    qty_dot_params = sum(param.count('.') for param in param_names)

    # Attribute 2: Number of "-" signs in the parameters
    qty_hyphen_params = sum(param.count('-') for param in param_names)

    # Attribute 3: Number of "_" signs in the parameters
    qty_underline_params = sum(param.count('_') for param in param_names)

    # Attribute 4: Number of "/" signs in the parameters
    qty_slash_params = sum(param.count('/') for param in param_names)

    # Attribute 5: Number of "?" signs in the parameters
    qty_questionmark_params = sum(param.count('?') for param in param_names)

    # Attribute 6: Number of "%" signs in the parameters
    qty_percent_params = sum(param.count('%') for param in param_names)

    # Return the extracted features
    return {
        'qty_dot_params': qty_dot_params,
        'qty_hyphen_params': qty_hyphen_params,
        'qty_underline_params': qty_underline_params,
        'qty_slash_params': qty_slash_params,
        'qty_questionmark_params': qty_questionmark_params,
        'qty_percent_params': qty_percent_params
    }


def email_urlshorten(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the domain from the URL
    domain = parsed_url.netloc

    # Attribute: Check if email is present in the URL
    email_in_url = 1 if '@' in url else 0

    # Attribute: Check if the URL is shortened (assuming common URL shorteners)
    common_shorteners = ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly']  # Add more as needed
    url_shortened = 1 if domain in common_shorteners else 0

    # Return the extracted features
    return {
        'email_in_url': email_in_url,
        'url_shortened': url_shortened
    }

def extract_all_features(url):
    parsed_url = urlparse(url)
    # Extract URL-based features
    url_features = extract_additional_url_features(url)

    # Extract Domain-based features
    domain_features = extract_additional_domain_features(url)

    # Extract Page-based features
    path_features = extract_additional_path_features(url)

    # Extract File-based feature
    file_feature = extract_file_features(url)

    # Extract Params-based features
    params_features = extract_additional_params_features(url)

    # Extract Additional Features
    additional_features = email_urlshorten(url)

    # Combine all features
    all_features = {**url_features, **domain_features, **path_features, **file_feature , **params_features, **additional_features}

    return all_features
'''
import numpy as np
from urllib.parse import urlparse, parse_qs
import joblib
import socket

def is_ip_address(domain):
    try:
        socket.inet_aton(domain)
        return True
    except socket.error:
        return False

def extract_additional_url_features(url):
    parsed_url = urlparse(url)

    return {
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
        'qty_tld_url': len(parsed_url.netloc.split('.')[-1]), 
        'length_url': len(url)
    }

def extract_additional_domain_features(url):
    # Parse the URL to get the domain
    domain = urlparse(url).netloc

    if not domain:
        return {
            'qty_dot_domain': -1,
            'qty_hyphen_domain': -1,
            'qty_underline_domain': -1,
            'qty_at_domain': -1,
            'qty_vowels_domain': -1,
            'domain_in_ip': -1,
            'server_client_domain': -1
        }

    return {
        'qty_dot_domain': domain.count('.'),
        'qty_hyphen_domain': domain.count('-'),
        'qty_underline_domain': domain.count('_'),
        'qty_at_domain': domain.count('@'),
        'qty_vowels_domain': sum(1 for char in domain if char.lower() in "aeiou"),
        'domain_in_ip': 1 if is_ip_address(domain) else 0,
        'server_client_domain': 1 if domain.startswith("www.") else 0
    }

def extract_additional_path_features(url):
    # Parse the URL to get the path
    path = urlparse(url).path

    if not path:
        return {
            'qty_dot_directory': -1,
            'qty_hyphen_directory': -1,
            'qty_underline_directory': -1,
            'qty_percent_directory': -1,
            'directory_length': -1
        }

    return {
        'qty_dot_directory': path.count('.'),
        'qty_hyphen_directory': path.count('-'),
        'qty_underline_directory': path.count('_'),
        'qty_percent_directory': path.count('%'),
        'directory_length': len(path)
    }

def extract_file_features(url):
    # Parse the URL to get the path
    path = urlparse(url).path

    if not path:
        return {
            'file_length': -1
        }

    # Extract the file name from the path
    file_name = path.split('/')[-1]

    # Attribute: Length of the file name
    file_length = len(file_name)

    return {
        'file_length': file_length
    }

def extract_additional_params_features(url):
    # Parse the URL to get the query parameters
    query_params = urlparse(url).query

    if not query_params:
        return {
            'qty_dot_params': -1,
            'qty_hyphen_params': -1,
            'qty_underline_params': -1,
            'qty_slash_params': -1,
            'qty_questionmark_params': -1,
            'qty_percent_params': -1
        }

    # Extract parameter names from the query string
    param_names = parse_qs(query_params).keys()

    return {
        'qty_dot_params': sum(param.count('.') for param in param_names),
        'qty_hyphen_params': sum(param.count('-') for param in param_names),
        'qty_underline_params': sum(param.count('_') for param in param_names),
        'qty_slash_params': sum(param.count('/') for param in param_names),
        'qty_questionmark_params': sum(param.count('?') for param in param_names),
        'qty_percent_params': sum(param.count('%') for param in param_names)
    }

def email_urlshorten(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the domain from the URL
    domain = parsed_url.netloc

    if not domain:
        return {
            'email_in_url': -1,
            'url_shortened': -1
        }

    return {
        'email_in_url': 1 if '@' in url else 0,
        'url_shortened': 1 if domain in ['bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly'] else 0
    }

def extract_all_features(url):
    # Extract URL-based features
    url_features = extract_additional_url_features(url)

    # Extract Domain-based features
    domain_features = extract_additional_domain_features(url)

    # Extract Page-based features
    path_features = extract_additional_path_features(url)

    # Extract File-based feature
    file_feature = extract_file_features(url)

    # Extract Params-based features
    params_features = extract_additional_params_features(url)

    # Extract Additional Features
    additional_features = email_urlshorten(url)

    # Combine all features
    all_features = {**url_features, **domain_features, **path_features, **file_feature, **params_features, **additional_features}

    return all_features

# Example usage
'''url = "https://www.example.org/products/category1/productA/details?color=blue&size=medium"
features = extract_all_features(url)
print(features)
'''

def prediction(extracted_features):
    #loaded_pipeline = joblib.load('/content/tpot_xgbclassifier_pipeline.joblib')
    print(extracted_features.values(),len(extracted_features.values()))
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