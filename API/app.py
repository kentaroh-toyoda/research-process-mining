# import main Flask class and request object
from flask import Flask, request

# GET requests will be blocked
@app.route('/auto-process-discovery', methods=['POST'])
def auto_process_discovery():
    request_data = request.get_json()

    if request_data:
        if 'language' in request_data:
            language = request_data['language']

        if 'examples' in request_data:
            if (type(request_data['examples']) == list) and (len(request_data['examples']) > 0):
                example = request_data['examples'][0]

        if 'boolean_test' in request_data:
            boolean_test = request_data['boolean_test']

    return '''
           The language value is: {}
           The framework value is: {}
           The Python version is: {}
           The item at index 0 in the example list is: {}
           The boolean value is: {}'''.format(language, framework, python_version, example, boolean_test)