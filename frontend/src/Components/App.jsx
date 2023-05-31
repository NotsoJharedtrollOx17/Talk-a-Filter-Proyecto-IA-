import React, { useState } from 'react';
import '../Stylings/App.css';

function App() {
  const [posts, setPosts] = useState([]);
  const [subreddit, setSubreddit] = useState('');
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const fetchPosts = () => {
    clearPosts();
    setSubmitted(true);
    setLoading(true);
    fetch(`http://localhost:5000/posts/${subreddit}`)
      .then((res) => res.json())
      .then((data) => {
        setPosts(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error('Error fetching posts:', error);
        setLoading(false);
      });
  };

  const clearPosts = () => {
    setSubreddit('');
    setLoading(false);
    setSubmitted(false);
    setPosts([]);
  };

  return (
    <div className='App'>
      {!submitted ? (
        <div className='SearchScreen'>
          <div className='Title'>
            <h2>Reddit Posts via </h2>
            <h1>Talk-a-Filter</h1>
          </div>
          <div className='button-container'>
            <input
              id='subreddit-input'
              type='text'
              placeholder='Enter subreddit name'
              value={subreddit}
              onChange={(e) => setSubreddit(e.target.value)}
            />
            <button onClick={fetchPosts}>Get Posts</button>
            <button onClick={clearPosts}>Clear Posts</button>
          </div>
        </div>
      ) : (
        <div>
          <div>
            <meta charSet='UTF-8'></meta>
            <title>Back End Talk A Filter</title>
            <script src='https://kit.fontawesome.com/fff4cc2dad.js' crossOrigin='anonymous'></script>
            <link rel='stylesheet' href='../Stylings/App.css'></link>
          </div>
          <header>
            <div className='wrapper'>
              <div className='multi_color_border'></div>
              <div className='top_nav'>
                <div className='left'>
                  <div className='logo'>
                    <img src='../img/logo_TECT.png' alt='' />
                    <p>TALK A FILTER</p>
                  </div>
                </div>
                <div className='right'>
                  <ul>
                    <li>
                      <a href='#'><h6>Equipo 2</h6></a>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </header>
          <nav>
            <header>
              <div className='Barra-Contenedor'>
                <div className='bottom_nav'></div>
                <div className='Barra'>
                  <div className='user'>
                    <div className='avatar' style={{ backgroundImage: 'url(img/icon.jpg)' }}></div>
                    <div className='Name'>
                      <p>Anonimo</p>
                      <li>Debajo se muestran las publicaciones del subreddit elegido</li>
                    </div>
                  </div>
                </div>
              </div>
            </header>
          </nav>

          <section>
            <article>
              <div className='Contenedor'>
                <div className='Cont-Left'>
                  <div className='BarraPost'>
                    <div className='MenuPost'>
                      <ul>
                        <li>
                          <i className='fa-brands fa-hotjar'></i>
                          <a href='#'>POSTS</a>
                        </li>
                        <li>
                          <a href='#'>. . .</a>
                        </li>
                      </ul>
                    </div>
                  </div>
                  <div className='Post'>
                    <div className='Comentarios'>
                      <ul id='comments-list' className='comments-list'>
                        <div>
                          {loading}
                          {!loading && (
                            <div>
                              {' '}
                              {posts.map((post) => (
                                <li key={post.id} className='post'>
                                  <div className='comment-main-level'>
                                    <div className='comment-avatar'>
                                      <img src='http://i9.photobucket.com/albums/a88/creaticode/avatar_1_zps8e1c80cd.jpg' alt='' />
                                    </div>
                                    <div className='comment-box'>
                                      <div className='comment-head'>
                                        <h6 className='comment-name by-author'>
                                          <a href='http://creaticode.com/blog'>{post.author}</a>
                                        </h6>
                                        <span>hace 20 minutos</span>
                                        <i className='fa fa-reply'></i>
                                        <i className='fa fa-heart'></i>
                                      </div>
                                      <div className='comment-content'>
                                        <h3>{post.title}</h3>
                                        <span>{post.text}</span>
                                      </div>
                                    </div>
                                  </div>

                                  <ul className='comment-container'>
                                    {post.comments.map((comment) => (
                                      <ul key={comment.id} className='comment'>
                                        <ul className='comments-list reply-list'>
                                          <li>
                                            <div className='comment-avatar'>
                                              <img src='http://i9.photobucket.com/albums/a88/creaticode/avatar_2_zps7de12f8b.jpg' alt='' />
                                            </div>
                                            <div className='comment-box'>
                                              <div className='comment-head'>
                                                <h6 className='comment-name'>
                                                  <a href='http://creaticode.com/blog'>{comment.author}</a>
                                                </h6>
                                                <span>hace 10 minutos</span>
                                                <i className='fa fa-reply'></i>
                                                <i className='fa fa-heart'></i>
                                              </div>
                                              <div className='comment-content'>
                                                <span>{comment.text}</span>
                                              </div>
                                            </div>
                                          </li>
                                        </ul>
                                      </ul>
                                    ))}
                                  </ul>
                                </li>
                              ))}
                            </div>
                          )}
                        </div>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </article>

            <aside>
              <div className='Cont-Right'>
                <div className='Community'>
                  <div className='linea2'>
                  </div>
                  <div className='CreatePost'>
                    <button className='Button-Post' role='button' onClick={clearPosts}>
                      Search new Filtered Subreddit
                    </button>
                  </div>
                  <div className='linea3'></div>
                </div>

                <div className='Moderator'></div>
              </div>
            </aside>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;