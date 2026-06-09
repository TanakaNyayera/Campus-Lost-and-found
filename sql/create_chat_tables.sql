-- MSU Lost & Found — Support Chat Tables
-- Notes:
-- 1) The Flask app expects these tables/columns.
-- 2) Run these statements against your MySQL database (msu_lost_found).

CREATE TABLE IF NOT EXISTS support_rooms (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id INT NOT NULL,
  room_type VARCHAR(50) NOT NULL,
  created_at DATETIME NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_support_rooms_user_room_type (user_id, room_type),
  CONSTRAINT fk_support_rooms_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  room_id BIGINT UNSIGNED NOT NULL,
  sender_user_id INT NOT NULL,
  message TEXT NOT NULL,
  -- Optional attachment (for clarification)
  attachment_type VARCHAR(20) NULL,
  attachment_path TEXT NULL,
  created_at DATETIME NOT NULL,
  PRIMARY KEY (id),
  KEY idx_chat_messages_room_id_id (room_id, id),
  CONSTRAINT fk_chat_messages_room
    FOREIGN KEY (room_id) REFERENCES support_rooms(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_chat_messages_sender
    FOREIGN KEY (sender_user_id) REFERENCES users(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Optional: if you want attachments (images/files) later, extend chat_messages with:
--   attachment_type VARCHAR(20) NULL,
--   attachment_path TEXT NULL
-- and update UI + backend to accept uploads.
