import React, { useState } from 'react';

function App() {
  const [posts, setPosts] = useState([]);
  const [subreddit, setSubreddit] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const fetchPosts = () => {
    fetch(`http://localhost:5000/posts/${subreddit}`)
      .then((res) => res.json())
      .then((data) => {
        setPosts(data);
      })
      .catch((error) => console.log(error));
  };

  const clearPosts = () => {
    setPosts([]);
  };
  
  return (
    <div>
      <h1>Reddit Posts</h1>
      <input
        type="text"
        placeholder="Enter subreddit name"
        value={subreddit}
        onChange={(e) => setSubreddit(e.target.value)}
      />
      <button onClick={fetchPosts}>Fetch Posts</button>
      <button onClick={clearPosts}>Clear Posts</button>
      <ul>
        {posts.map((post) => (
          <li key={post.id}>
            <h3>{post.title}</h3>
            <p>{post.text}</p>
            <p>Author: {post.author}</p>
            <p>Score: {post.score}</p>
            <a href={post.url}>Link to post</a>
            <ul>
              {post.comments.map((comment) => (
                <li key={comment.id}>
                  <p>{comment.text}</p>
                  <p>Author: {comment.author}</p>
                  <p>Score: {comment.score}</p>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;