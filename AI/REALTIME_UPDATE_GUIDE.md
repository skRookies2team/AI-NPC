# 실시간 RAG 업데이트 가이드

## 개요

소설 게임에서 NPC가 실시간으로 변화하는 소설 상황을 학습할 수 있도록, RAG 시스템에 새로운 내용을 추가하는 기능을 구현했습니다.

## 기능 설명

게임 진행 중 새로운 이벤트, 스토리 전개, 플레이어의 선택에 따른 변화 등이 발생할 때, 이러한 내용을 즉시 RAG 벡터 데이터베이스에 추가하여 NPC가 최신 상황을 반영하여 대화할 수 있습니다.

## API 엔드포인트

### 1. `/api/ai/update` - 텍스트 직접 추가

실시간으로 발생한 소설 상황을 텍스트로 직접 추가합니다.

**요청:**
```json
POST /api/ai/update
Content-Type: application/json

{
  "session_id": "your_session_id",
  "content": "플레이어가 마법의 검을 획득했습니다. 검은 빛을 내며 강력한 힘을 발산합니다...",
  "metadata": {
    "chapter": 5,
    "event_type": "item_acquired",
    "player_action": "explore_dungeon"
  }
}
```

**응답:**
```json
{
  "status": "updated",
  "session_id": "your_session_id",
  "chunks_added": 2,
  "message": "새로운 소설 내용 2개 청크가 성공적으로 추가되었습니다."
}
```

### 2. `/api/ai/update-file` - 파일로 대량 추가

대량의 새로운 소설 내용을 파일로 업로드하여 추가합니다.

**요청:**
```
POST /api/ai/update-file
Content-Type: multipart/form-data

session_id: your_session_id
file: new_chapter.txt
```

**응답:**
```json
{
  "status": "updated",
  "session_id": "your_session_id",
  "chunks_added": 15,
  "message": "파일에서 15개 청크가 성공적으로 추가되었습니다."
}
```

## 사용 예시

### Python 예시

```python
import requests

# 실시간 이벤트 추가
def update_novel_event(session_id, event_text, metadata=None):
    url = "http://localhost:8000/api/ai/update"
    payload = {
        "session_id": session_id,
        "content": event_text,
        "metadata": metadata or {}
    }
    response = requests.post(url, json=payload)
    return response.json()

# 사용 예시
update_novel_event(
    session_id="sess_123456",
    event_text="""
    플레이어가 마을의 용사 길드에 가입했습니다. 
    길드장은 플레이어에게 첫 번째 퀘스트를 부여했습니다.
    "동쪽 숲에 나타난 고블린들을 처치해주시오."
    """,
    metadata={
        "chapter": 3,
        "event_type": "guild_join",
        "quest_available": True
    }
)
```

### JavaScript/TypeScript 예시

```typescript
// 실시간 이벤트 추가
async function updateNovelEvent(
  sessionId: string, 
  eventText: string, 
  metadata: Record<string, any> = {}
) {
  const response = await fetch('http://localhost:8000/api/ai/update', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      content: eventText,
      metadata: metadata
    })
  });
  return await response.json();
}

// 사용 예시
await updateNovelEvent(
  'sess_123456',
  `플레이어가 드래곤과의 전투에서 승리했습니다.
   드래곤의 보물창고에서 전설의 반지를 발견했습니다.`,
  {
    chapter: 10,
    event_type: 'battle_victory',
    boss_defeated: 'ancient_dragon'
  }
);
```

## 게임 통합 시나리오

### 시나리오 1: 플레이어 선택에 따른 분기

```python
# 플레이어가 선택지를 선택했을 때
def on_player_choice(session_id, choice_result):
    if choice_result == "rescue_villager":
        event_text = """
        플레이어는 위험을 무릅쓰고 마을 사람들을 구했습니다.
        마을 사람들은 감사하며 플레이어에게 특별한 보상을 주었습니다.
        플레이어의 평판이 크게 올라갔습니다.
        """
    elif choice_result == "ignore_villager":
        event_text = """
        플레이어는 마을 사람들을 무시하고 길을 계속 갔습니다.
        마을 사람들은 실망했고, 플레이어의 평판이 떨어졌습니다.
        """
    
    update_novel_event(
        session_id,
        event_text,
        metadata={"choice_id": choice_result, "timestamp": get_current_time()}
    )
```

### 시나리오 2: 퀘스트 완료

```python
def on_quest_complete(session_id, quest_name, reward):
    event_text = f"""
    플레이어는 '{quest_name}' 퀘스트를 완료했습니다.
    퀘스트 보상으로 {reward}를 획득했습니다.
    새로운 지역이 열렸습니다.
    """
    
    update_novel_event(
        session_id,
        event_text,
        metadata={
            "quest_name": quest_name,
            "reward": reward,
            "new_area_unlocked": True
        }
    )
```

### 시나리오 3: 주요 스토리 진행

```python
def on_story_progress(session_id, story_event):
    # 주요 스토리 이벤트 발생 시
    update_novel_event(
        session_id,
        story_event.description,
        metadata={
            "story_point": story_event.point_id,
            "chapter": story_event.chapter,
            "importance": "major"
        }
    )
```

## 작동 원리

1. **초기 학습**: `/api/ai/train`으로 기본 소설 내용을 학습
2. **실시간 업데이트**: 게임 진행 중 `/api/ai/update` 또는 `/api/ai/update-file`로 새 내용 추가
3. **자동 인덱싱**: 추가된 내용은 자동으로 청킹되고 벡터화되어 기존 벡터 DB에 추가
4. **즉시 반영**: 추가된 내용은 다음 대화부터 자동으로 검색되어 반영됨

## 주의사항

1. **세션 존재 확인**: `/api/ai/update`를 사용하기 전에 반드시 `/api/ai/train`으로 초기 학습이 완료되어 있어야 합니다.

2. **메타데이터 활용**: 
   - 메타데이터를 통해 나중에 특정 이벤트나 챕터의 내용을 필터링하거나 관리할 수 있습니다.
   - 예: `chapter`, `event_type`, `player_action`, `timestamp` 등

3. **성능 고려**:
   - 매우 짧은 텍스트를 너무 자주 추가하면 오버헤드가 발생할 수 있습니다.
   - 여러 작은 이벤트를 모아서 한 번에 추가하는 것을 권장합니다.

4. **내용 품질**:
   - 추가하는 내용은 기존 소설의 스타일과 일관성을 유지하는 것이 좋습니다.
   - 명확하고 문맥이 있는 문장으로 작성하면 더 좋은 검색 결과를 얻을 수 있습니다.

## 벡터 DB 상태 확인

추가된 내용이 제대로 반영되었는지 확인하려면:

1. NPC와 대화를 시도하여 새로운 상황을 언급하는지 확인
2. PostgreSQL에서 직접 벡터 DB를 조회 (고급 사용자용)

## 예상 시나리오

게임 진행 예시:
1. **초기**: 기본 소설 1-3장 학습 → NPC가 초반 스토리 기반으로 대화
2. **플레이어 진행**: 4-5장 이벤트 추가 → NPC가 최신 상황 반영
3. **플레이어 선택**: 분기점 선택 → 선택 결과 추가 → NPC가 선택에 따른 반응
4. **계속 진행**: 6-7장 추가 → NPC가 계속 업데이트된 스토리 반영

이를 통해 NPC는 항상 최신 소설 상황을 알고 있는 것처럼 대화할 수 있습니다!










