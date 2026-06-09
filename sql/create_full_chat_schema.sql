-- MSU Lost & Found — Full Chat Schema Helper
--
-- Purpose:
-- 1) Creates the tables required by the Flask chat endpoints.
-- 2) Intended to be safe to run multiple times.
--
-- Run against your MySQL database (msu_lost_found).
--
-- Required pre-existing tables (for foreign keys):
--   - users(id)
--
-- If your `users` table uses a different name/PK, adjust accordingly.

SET FOREIGN_KEY_CHECKS=0;

-- Rooms: one per user for user<->admin support chat
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

-- Messages within a room
CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  room_id BIGINT UNSIGNED NOT NULL,
  sender_user_id INT NOT NULL,
  message TEXT NOT NULL,
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

SET FOREIGN_KEY_CHECKS=1;

