from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    if username == 'itublg' and password == '1234':
        # Successful login
        return render_template('index.html')
    else:
        # Invalid credentials, redirect back to the login page
        return redirect('/')

@app.route('/tsne')
def tsne():
    return render_template('tsne.html')

@app.route('/word2vec')
def word2vec():
    return render_template('word2vec.html')

@app.route('/kmeans')
def kmeans():
    return render_template('kmeans_page.html')



if __name__ == '__main__':
    app.run(debug=True)