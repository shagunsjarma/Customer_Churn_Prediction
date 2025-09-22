from flask import request, jsonify
from functools import wraps
import time
import logging

def log_request_info(f):
    """
    Decorator to log API request information
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        # Log request details
        logging.info(f"API Request: {request.method} {request.path}")
        logging.info(f"Request IP: {request.remote_addr}")
        logging.info(f"Request Headers: {dict(request.headers)}")
        
        if request.is_json:
            logging.info(f"Request Body: {request.get_json()}")
        
        # Execute the function
        response = f(*args, **kwargs)
        
        # Log response time
        end_time = time.time()
        response_time = end_time - start_time
        logging.info(f"Response Time: {response_time:.3f} seconds")
        
        return response
    return decorated_function

def validate_content_type(f):
    """
    Decorator to validate content type for POST requests
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'POST':
            if not request.is_json:
                return jsonify({
                    'error': 'Content-Type must be application/json'
                }), 400
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(max_requests=100, window=3600):
    """
    Simple rate limiting decorator
    Note: In production, use Redis or similar for distributed rate limiting
    """
    def decorator(f):
        request_counts = {}
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old entries
            request_counts[client_ip] = [
                req_time for req_time in request_counts.get(client_ip, [])
                if current_time - req_time < window
            ]
            
            # Check rate limit
            if len(request_counts.get(client_ip, [])) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'limit': max_requests,
                    'window': window
                }), 429
            
            # Add current request
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            request_counts[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def handle_errors(f):
    """
    Global error handler decorator
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Unhandled error in {f.__name__}: {str(e)}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            }), 500
    return decorated_function
