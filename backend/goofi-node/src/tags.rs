//! The closed tag vocabulary. The engine and the bundle are facets the palette derives, never tags.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Tag {
    Input,
    Output,
    Generator,
    Transform,
    Analysis,
    Control,
    Image,
    Text,
    Midi,
    Eeg,
    Cardio,
    Motion,
    Music,
    Ml,
    Connectivity,
    Simulation,
}

impl Tag {
    pub const ALL: &'static [Tag] = &[
        Tag::Input,
        Tag::Output,
        Tag::Generator,
        Tag::Transform,
        Tag::Analysis,
        Tag::Control,
        Tag::Image,
        Tag::Text,
        Tag::Midi,
        Tag::Eeg,
        Tag::Cardio,
        Tag::Motion,
        Tag::Music,
        Tag::Ml,
        Tag::Connectivity,
        Tag::Simulation,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Tag::Input => "input",
            Tag::Output => "output",
            Tag::Generator => "generator",
            Tag::Transform => "transform",
            Tag::Analysis => "analysis",
            Tag::Control => "control",
            Tag::Image => "image",
            Tag::Text => "text",
            Tag::Midi => "midi",
            Tag::Eeg => "eeg",
            Tag::Cardio => "cardio",
            Tag::Motion => "motion",
            Tag::Music => "music",
            Tag::Ml => "ml",
            Tag::Connectivity => "connectivity",
            Tag::Simulation => "simulation",
        }
    }

    pub fn parse(s: &str) -> Option<Tag> {
        Tag::ALL.iter().copied().find(|t| t.as_str() == s)
    }

    pub fn vocabulary() -> String {
        Tag::ALL.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(", ")
    }
}
