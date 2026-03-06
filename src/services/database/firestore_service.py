from google.cloud import firestore
from ...utils.config import get_settings
from ...utils.logger import setup_logger
from google.cloud.firestore_v1.base_query import FieldFilter

settings = get_settings()
logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Collection name maps
# Titles:  rio_titles_theme1/2/3
# Stories: rio_stories_theme1/2/3
# Activities: activities_v1 (unchanged, tagged with story_id)
# ---------------------------------------------------------------------------

_TITLE_COLLECTIONS = {
    "theme1": "rio_titles_theme1",
    "theme2": "rio_titles_theme2",
    "theme3": "rio_titles_theme3",
}

_STORY_COLLECTIONS = {
    "theme1": "rio_stories_theme1",
    "theme2": "rio_stories_theme2",
    "theme3": "rio_stories_theme3",
}


class FirestoreService:
    def __init__(self):
        self._db = None

    @property
    def db(self):
        if self._db is None:
            project = settings.GOOGLE_CLOUD_PROJECT
            database = settings.FIRESTORE_DATABASE
            logger.info(f"Initializing Firestore: project={project} database={database}")
            if settings.GOOGLE_APPLICATION_CREDENTIALS:
                self._db = firestore.Client.from_service_account_json(
                    json_credentials_path=settings.GOOGLE_APPLICATION_CREDENTIALS,
                    project=project,
                    database=database,
                )
            else:
                self._db = firestore.Client(project=project, database=database)
        return self._db

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _title_collection(theme: str) -> str:
        return _TITLE_COLLECTIONS.get(theme, "rio_titles_theme1")

    @staticmethod
    def _story_collection(theme: str) -> str:
        return _STORY_COLLECTIONS.get(theme, "rio_stories_theme1")

    # ------------------------------------------------------------------
    # Activities  (activities_v1 — unchanged)
    # ------------------------------------------------------------------

    async def check_if_activity_exists(self, story_id: str, activity_type: str):
        try:
            query = (
                self.db.collection("activities_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .where(filter=FieldFilter("type", "==", activity_type))
                .limit(1)
            )
            doc = next(query.stream(), None)
            return doc.to_dict() if doc else None
        except Exception as e:
            logger.error(f"check_if_activity_exists failed: {e}")
            return None

    async def save_activity(self, story_id: str, activity_type: str, activity_data) -> None:
        """
        Saves an activity to activities_v1 and sets activities.{type}='ready'
        on the story document. Story is looked up across all theme collections
        using story_id as the document ID (O(1) direct reads, no query).
        """
        try:
            # Deduplicate
            dup_q = (
                self.db.collection("activities_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .where(filter=FieldFilter("type", "==", activity_type))
                .limit(1)
            )
            if next(dup_q.stream(), None):
                logger.info(f"Activity {activity_type} already exists for story {story_id}")
                return

            # Find story ref — story_id IS the doc ID in theme collections
            story_ref = None
            for col in _STORY_COLLECTIONS.values():
                ref = self.db.collection(col).document(story_id)
                if ref.get().exists:
                    story_ref = ref
                    break

            if story_ref is None:
                logger.error(f"Story {story_id} not found in any theme collection")
                raise ValueError(f"Story {story_id} not found")

            if isinstance(activity_data, list):
                data_to_save = {"items": activity_data}
            elif isinstance(activity_data, dict):
                data_to_save = activity_data
            else:
                data_to_save = {"data": activity_data}

            batch = self.db.batch()
            activity_ref = self.db.collection("activities_v1").document()
            batch.set(activity_ref, {
                **data_to_save,
                "story_id": story_id,
                "type": activity_type,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            batch.update(story_ref, {
                f"activities.{activity_type}": "ready",
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            batch.commit()
            logger.info(f"Saved activity {activity_type} for story {story_id}")
        except Exception as e:
            logger.error(f"save_activity failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Stories  (rio_stories_theme{N} — story_id as doc ID)
    # ------------------------------------------------------------------

    async def get_story(self, story_id: str, theme: str | None = None) -> dict | None:
        """
        Direct doc lookup (story_id = doc ID). Searches the given theme collection
        first; if not found (or no theme given) tries all 3.
        """
        try:
            cols = [self._story_collection(theme)] if theme else list(_STORY_COLLECTIONS.values())
            for col in cols:
                doc = self.db.collection(col).document(story_id).get()
                if doc.exists:
                    return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"get_story failed: {e}")
            return None

    async def save_story(self, story_id: str, story: dict, theme: str) -> None:
        """Saves/upserts story to rio_stories_theme{N} with story_id as document ID."""
        try:
            col = self._story_collection(theme)
            payload = {
                **story,
                "story_id": story_id,
                "theme": theme,
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            self.db.collection(col).document(story_id).set(payload, merge=True)
            logger.info(f"[Firestore] Story saved: {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story failed: {e}")
            raise

    async def save_story_image(
        self, story_id: str, image_url: str, generation_prompt: str, theme: str
    ) -> None:
        """Updates image_url on the story doc and logs metadata in story_images_v1."""
        try:
            col = self._story_collection(theme)
            self.db.collection(col).document(story_id).update({
                "image_url": image_url,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            self.db.collection("story_images_v1").document().set({
                "story_id": story_id,
                "image_url": image_url,
                "generation_prompt": generation_prompt,
                "theme": theme,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] image_url saved on {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story_image failed: {e}")
            raise

    async def save_story_audio(
        self, story_id: str, audio_url: str, language: str, voice: str, theme: str
    ) -> None:
        """Updates audio_url on the story doc and logs metadata in story_audio_v1."""
        try:
            col = self._story_collection(theme)
            self.db.collection(col).document(story_id).update({
                "audio_url": audio_url,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            self.db.collection("story_audio_v1").document().set({
                "story_id": story_id,
                "audio_url": audio_url,
                "language": language,
                "voice": voice,
                "theme": theme,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] audio_url saved on {col}/{story_id}")
        except Exception as e:
            logger.error(f"save_story_audio failed: {e}")
            raise

    # ------------------------------------------------------------------
    # WF1 session log  (story_topics_v1 — reference only, not the title lib)
    # ------------------------------------------------------------------

    async def save_story_topics(self, story_id: str, topics: list) -> None:
        try:
            self.db.collection("story_topics_v1").document().set({
                "story_id": story_id,
                "topics": topics,
                "selected_topic": None,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] Topics session saved for story_id={story_id}")
        except Exception as e:
            logger.error(f"save_story_topics failed: {e}")
            raise

    async def get_story_topics(self, story_id: str) -> dict | None:
        try:
            q = (
                self.db.collection("story_topics_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .limit(1)
            )
            docs = list(q.stream())
            return docs[0].to_dict() if docs else None
        except Exception as e:
            logger.error(f"get_story_topics failed: {e}")
            return None

    async def set_selected_topic(self, story_id: str, selected_topic: dict) -> None:
        try:
            q = (
                self.db.collection("story_topics_v1")
                .where(filter=FieldFilter("story_id", "==", story_id))
                .limit(1)
            )
            docs = list(q.stream())
            if not docs:
                raise ValueError(f"No topics doc for story_id={story_id}")
            docs[0].reference.update({
                "selected_topic": selected_topic,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
        except Exception as e:
            logger.error(f"set_selected_topic failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Title Library  (rio_titles_theme{N})
    # Doc ID: {age_norm}__{lang}__{filter_value_norm}
    # ------------------------------------------------------------------

    @staticmethod
    def _library_doc_id(age: str, lang: str, filter_value: str) -> str:
        """Deterministic Firestore-safe doc ID (theme is encoded by the collection name)."""
        import re
        norm = re.sub(r"[^a-z0-9]+", "_", filter_value.lower()).strip("_")
        return f"{age.replace('-', '_')}__{lang}__{norm}"

    async def get_title_library_entry(
        self, theme: str, age: str, lang: str, filter_value: str
    ) -> list | None:
        """Returns cached titles from the theme title collection, or None on miss."""
        try:
            col = self._title_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            doc = self.db.collection(col).document(doc_id).get()
            if doc.exists:
                titles = doc.to_dict().get("titles", [])
                logger.info(f"[Firestore] Cache hit: {col}/{doc_id} ({len(titles)} titles)")
                return titles
            return None
        except Exception as e:
            logger.error(f"get_title_library_entry failed: {e}")
            return None

    async def save_title_library_entry(
        self,
        theme: str,
        age: str,
        lang: str,
        filter_type: str,
        filter_value: str,
        titles: list,
    ) -> None:
        """Upserts generated titles into the theme title collection."""
        try:
            col = self._title_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            self.db.collection(col).document(doc_id).set({
                "theme":        theme,
                "age":          age,
                "language":     lang,
                "filter_type":  filter_type,
                "filter_value": filter_value,
                "titles":       titles,
                "created_at":   firestore.SERVER_TIMESTAMP,
            })
            logger.info(f"[Firestore] Titles saved: {col}/{doc_id} ({len(titles)} titles)")
        except Exception as e:
            logger.error(f"save_title_library_entry failed: {e}")
            raise

    async def update_title_story_id(
        self,
        theme: str,
        age: str,
        lang: str,
        filter_value: str,
        title_text: str,
        story_id: str,
    ) -> None:
        """Patches story_id onto a specific title entry in the theme title collection."""
        try:
            col = self._title_collection(theme)
            doc_id = self._library_doc_id(age, lang, filter_value)
            doc_ref = self.db.collection(col).document(doc_id)
            doc = doc_ref.get()
            if not doc.exists:
                logger.warning(f"[Firestore] Title doc not found: {col}/{doc_id}")
                return
            titles = list(doc.to_dict().get("titles", []))
            for t in titles:
                if t.get("title") == title_text:
                    t["story_id"] = story_id
                    doc_ref.update({"titles": titles, "updated_at": firestore.SERVER_TIMESTAMP})
                    logger.info(f"[Firestore] story_id={story_id} patched in {col}/{doc_id}")
                    return
            logger.warning(f"[Firestore] Title '{title_text}' not found in {col}/{doc_id}")
        except Exception as e:
            logger.error(f"update_title_story_id failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_workflow_status(self, story_id: str) -> dict:
        try:
            story = await self.get_story(story_id)
            if not story:
                return {"story_id": story_id, "status": "not_found"}
            return {
                "story_id":       story_id,
                "theme":          story.get("theme"),
                "wf2_story":      "completed" if story.get("story_text") else "pending",
                "wf3_image":      "completed" if story.get("image_url") else "pending",
                "wf4_audio":      "completed" if story.get("audio_url") else "pending",
                "wf5_activities": story.get("activities", {}),
            }
        except Exception as e:
            logger.error(f"get_workflow_status failed: {e}")
            return {"story_id": story_id, "status": "error", "error": str(e)}
