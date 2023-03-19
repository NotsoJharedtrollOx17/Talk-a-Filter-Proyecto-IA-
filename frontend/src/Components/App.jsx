import React, { useState, useEffect } from "react";
import '../Stylings/App.css'

const App = () => {
  const [subreddit, setSubreddit] = useState("");
  const [posts, setPosts] = useState([]);

  const fetchPosts = async () => {
    const response = await fetch(`/api/posts?subreddit=${subreddit}`);
    const data = await response.json();
    setPosts(data);
  };

  const clearPosts = () => {
    setPosts([]);
  };

  return (
    <div>
      <select value={subreddit} onChange={(e) => setSubreddit(e.target.value)}>
        <option value="">Select a subreddit</option>
        <option value="aww">aww</option>
        <option value="gifs">gifs</option>
        <option value="mildlyinteresting">mildlyinteresting</option>
        <option value="pics">pics</option>
        <option value="todayilearned">todayilearned</option>
      </select>
      <button onClick={fetchPosts}>Fetch posts</button>
      <button onClick={clearPosts}>Clear posts</button>
      {posts.map((post) => (
        <Post key={post.id} post={post} />
      ))}
    </div>
  );
};

const Post = ({ post }) => {
  const [comments, setComments] = useState([]);

  useEffect(() => {
    const fetchComments = async () => {
      const response = await fetch(`/api/comments?post_id=${post.id}`);
      const data = await response.json();
      setComments(data);
    };

    fetchComments();
  }, [post.id]);

  return (
    <div>
      <h2>{post.title}</h2>
      <p>Author: {post.author}</p>
      <p>Score: {post.score}</p>
      <p>
        <a href={post.url}>Link</a>
      </p>
      <p>{post.text}</p>
      <h3>Comments:</h3>
      {comments.map((comment) => (
        <Comment key={comment.id} comment={comment} />
      ))}
    </div>
  );
};

const Comment = ({ comment }) => {
  return (
    <div>
      <p>Author: {comment.author}</p>
      <p>Score: {comment.score}</p>
      <p>{comment.text}</p>
    </div>
  );
};

export default App;

