-- MSU Lost & Found - Handling Gateway
--
-- Creates delivery/return tasks for approved claims.

CREATE TABLE IF NOT EXISTS handling_tasks (
  id INT NOT NULL AUTO_INCREMENT,
  claim_id INT NOT NULL,
  match_id INT NOT NULL,
  lost_item_id INT NOT NULL,
  found_item_id INT NOT NULL,
  user_id INT NOT NULL,
  agent_id INT NULL,
  destination VARCHAR(255) NULL,
  status ENUM('pending','accepted','delivered','failed','user_confirmed','user_disputed') NOT NULL DEFAULT 'pending',
  agent_report TEXT NULL,
  confirmation_token VARCHAR(128) NULL,
  accepted_at DATETIME NULL,
  completed_at DATETIME NULL,
  user_confirmed_at DATETIME NULL,
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  PRIMARY KEY (id),
  UNIQUE KEY uq_handling_tasks_claim (claim_id),
  UNIQUE KEY uq_handling_tasks_confirmation_token (confirmation_token),
  KEY idx_handling_tasks_status (status),
  KEY idx_handling_tasks_agent (agent_id),
  CONSTRAINT fk_handling_tasks_claim
    FOREIGN KEY (claim_id) REFERENCES claims(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_handling_tasks_match
    FOREIGN KEY (match_id) REFERENCES matches(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_handling_tasks_lost
    FOREIGN KEY (lost_item_id) REFERENCES lost_items(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_handling_tasks_found
    FOREIGN KEY (found_item_id) REFERENCES found_items(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_handling_tasks_user
    FOREIGN KEY (user_id) REFERENCES users(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_handling_tasks_agent
    FOREIGN KEY (agent_id) REFERENCES users(id)
    ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Create gateway tasks for claims that were approved before this table existed.
INSERT INTO handling_tasks
(claim_id, match_id, lost_item_id, found_item_id, user_id, destination, status, created_at, updated_at)
SELECT c.id, c.match_id, m.lost_item_id, m.found_item_id, c.user_id,
       COALESCE(f.location_found, 'MSU Lost & Found Office'),
       'pending', NOW(), NOW()
FROM claims c
JOIN matches m ON c.match_id = m.id
JOIN found_items f ON m.found_item_id = f.id
LEFT JOIN handling_tasks ht ON ht.claim_id = c.id
WHERE c.status='approved' AND ht.id IS NULL;
