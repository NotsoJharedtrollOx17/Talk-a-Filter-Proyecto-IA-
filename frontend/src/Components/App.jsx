import React, { useState } from 'react';
import '../Stylings/App.css'

function App() {
  const [posts, setPosts] = useState([]);
  const [subreddit, setSubreddit] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchPosts = () => {
    clearPosts();
    setLoading(true);
    fetch(`http://localhost:5000/posts/${subreddit}`)
      .then((res) => res.json())
      .then((data) => {
        setPosts(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching posts:', error);
        setLoading(false);
      });
  };

  const clearPosts = () => {
    setSubreddit('');
    setLoading(false);
    setPosts([]);
  };
  
  return (
    <div className='App'>
      <div className='Title'>
        <h2>Reddit Posts via </h2>
        <h1>Talk-a-Filter</h1>
      </div>
      <div className='button-container'>
        <input
          id='subreddit-input'
          type="text"
          placeholder="Enter subreddit name"
          value={subreddit}
          onChange={(e) => setSubreddit(e.target.value)}
        />
        <button onClick={fetchPosts}>Get Posts</button>
        <button onClick={clearPosts}>Clear Posts</button>
      </div> 
      {loading && <p>Loading...</p>}
      {!loading && (
        <ul className='post-container'>
          {posts.map((post) => (
            <li key={post.id} className='post'>
              <h3>{post.title}</h3>
              <p>{post.text}</p>
              <p>Author: {post.author}</p>
              <p>Score: {post.score}</p>
              <a href={post.url}>Link to post</a>
              <ul className='comment-container'>
                {post.comments.map((comment) => (
                  <li key={comment.id} className='comment'>
                    <p>{comment.text}</p>
                    <p>Author: {comment.author}</p>
                    <p>Score: {comment.score}</p>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;